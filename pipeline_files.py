import asyncio
import csv
import json
import logging
import os
import subprocess
import sys
import time
import warnings

from multiprocessing import Process, Queue, Event
from queue import Empty

from utils import (
    DotConfigParser,
    render_and_embed,
    get_mtg_tags,
    extract_chords,
    extract_midi_features,
)

# shut up warnings about e.g. not-found instruments
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# MUST be on, else the first tensorflow process will allocate all vram, leaving none for others
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
_logger = logging.getLogger(__name__)
_logger.setLevel(31)  # bypass WARNING logging level

# TODO see if you can augment anything
# TODO integrate gigamidi dataset
# TODO do logger stuff

GLOBAL_TIMEOUT = 1  # seconds
SAMPLE_RATE = 16000  # see if this messes with anything
shutdown_event = Event()


def run_async(fn, *args, **kwargs):
    asyncio.run(fn(*args, **kwargs))


async def render_worker(
    embedding_gfn,
    file_paths_queue,
    mood_queue,
    genre_queue,
    chord_queue,
    failed_renders_queue,
    midi_queue,
):
    # imports MUST be here and not in top level else tensorflow multiprocessing will break
    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

    loader = MonoLoader(sampleRate=16000, resampleQuality=1)
    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=embedding_gfn, output="PartitionedCall:1"
    )
    while not shutdown_event.is_set():
        try:
            file_path = file_paths_queue.get(timeout=GLOBAL_TIMEOUT)
            # Render and embed audio
            try:
                async with asyncio.timeout(120):
                    prefix = os.path.dirname(file_path)
                    suffix = os.path.basename(file_path).split(".")[0]
                    audio_file = os.path.join(prefix, suffix + ".wav")
                    subprocess.run(
                        [
                            "fluidsynth",
                            "-q",
                            "-ni",
                            "models/FluidR3_GM.sf2",
                            file_path,
                            "-F",
                            audio_file,
                            "-r",
                            "16000",
                        ],
                        stderr=subprocess.DEVNULL,
                    )
                    if audio_file is None or not os.path.exists(audio_file):
                        return None
                    loader.configure(filename=audio_file, sampleRate=16000, resampleQuality=1)
                    audio = loader()
                    embeddings = embedding_model(audio)
                    print(embeddings)
            except Exception:
                print(f"Render worker exception on {file_path}")
                failed_renders_queue.put(file_path)
                continue
            # Put same embedding in both analysis queues
            mood_queue.put((file_path, embeddings))
            genre_queue.put((file_path, embeddings))
            chord_queue.put((file_path, audio_file))
            midi_queue.put(file_path)
        except Empty:
            continue


async def analysis_worker(
    type, model_gfn, metadata, input_queue, results_queue, idle_threshold
):
    # imports MUST be here and not in top level else tf multiprocessing will break
    from essentia.standard import TensorflowPredict2D

    max_num_tags = 5 if type == "mood" else 4
    tag_threshold = 0.02 if type == "mood" else 0.05
    last_active = time.time()
    model = TensorflowPredict2D(graphFilename=model_gfn)
    while not shutdown_event.is_set():
        try:
            file_path, embedding = input_queue.get(timeout=GLOBAL_TIMEOUT)
            last_active = time.time()

            try:
                async with asyncio.timeout(120):
                    tags = get_mtg_tags(
                        embedding,
                        model,
                        metadata,
                        max_num_tags=max_num_tags,
                        tag_threshold=tag_threshold,
                    )
            except TimeoutError:
                print(f"{type} worker timed out on {file_path}")
                tags = None

            result = {type: tags}

            results_queue.put((file_path, type, result))

        except Empty:
            # Check if worker should terminate due to inactivity
            if time.time() - last_active > idle_threshold:
                break
            continue


async def chord_worker(audio_queue, results_queue, idle_threshold):
    from chord_extractor.extractors import Chordino

    last_active = time.time()
    chord_estimator = Chordino()
    while not shutdown_event.is_set():
        try:
            file_path, audio_file_path = audio_queue.get(timeout=GLOBAL_TIMEOUT)
            last_active = time.time()
            try:
                async with asyncio.timeout(120):
                    features = extract_chords(audio_file_path, chord_estimator)
            except TimeoutError:
                print(f"Chord worker timed out on {file_path}")
                features = None
            results_queue.put((file_path, "chords", features))
        except Empty:
            if time.time() - last_active > idle_threshold:
                break
            continue


async def midi_worker(instrumentmap, midi_queue, results_queue, idle_threshold):
    last_active = time.time()
    with open(instrumentmap, "r") as csvf:
        csv_reader = csv.reader(csvf)
        ilist_data = [row for row in csv_reader]
    while not shutdown_event.is_set():
        try:
            file_path = midi_queue.get(timeout=GLOBAL_TIMEOUT)
            last_active = time.time()
            try:
                async with asyncio.timeout(120):
                    features = extract_midi_features(file_path, ilist_data)
            except TimeoutError:
                print(f"MIDI worker timed out on {file_path}")
                features = None
            results_queue.put((file_path, "midi", features))
        except Empty:
            if time.time() - last_active > idle_threshold:
                break
            continue


class DynamicPoolManager:
    def __init__(
        self, worker_func, input_queue, backlog_factor, min_processes=1, max_processes=8
    ):
        self.processes: list[Process] = []
        self.worker_func = worker_func
        self.input_queue = input_queue
        self.min_processes = min_processes
        self.max_processes = max_processes
        self.backlog_factor = backlog_factor

    def monitor_and_adjust(self, results_queue):
        # Calculate queue backlog
        backlog = self.input_queue.qsize()

        # Spawn new process if backlog too high and below max
        if (
            backlog > len(self.processes) * self.backlog_factor
            and len(self.processes) < self.max_processes
        ):
            p = Process(target=self.worker_func, args=(self.input_queue, results_queue))
            p.start()
            self.processes.append(p)

        # Clean up finished processes
        self.processes = [p for p in self.processes if p.is_alive()]

        # Ensure minimum number of processes
        # print(self.min_processes - len(self.processes))
        while len(self.processes) < self.min_processes:
            print("Starting worker")
            p = Process(target=self.worker_func, args=(self.input_queue, results_queue))
            p.start()
            self.processes.append(p)


def spin_up_render_worker(render_processes, render_args):
    p = Process(
        target=run_async,
        args=(render_worker, *render_args),
    )
    print("Spun up render worker")
    p.start()
    render_processes.append(p)


def cleanup_processes(process_list):
    """Gracefully terminate processes and wait for them to finish."""
    for process in process_list:
        if process.is_alive():
            process.terminate()

    # Give processes time to terminate gracefully
    for process in process_list:
        process.join(timeout=5)

    # Force kill any remaining processes
    for process in process_list:
        if process.is_alive():
            process.kill()
            process.join()


def main():
    # holdover from when i used argparse lmao
    config = DotConfigParser()
    config.read("config.cfg")
    args = config.parse_args()

    # Create queues for distributing work to render pool
    file_paths_queue = Queue()
    mood_queue = Queue()
    genre_queue = Queue()
    midi_queue = Queue()
    chord_queue = Queue()
    results_queue = Queue()
    failed_renders_queue = Queue()

    # Create fixed-size render pool - this time actually using it
    render_processes = []
    render_args = (
        args.emb_model,
        file_paths_queue,
        mood_queue,
        genre_queue,
        chord_queue,
        failed_renders_queue,
        midi_queue,
    )

    for _ in range(args.processes):
        spin_up_render_worker(render_processes, render_args)

    # Load files into queue
    with open(args.input_file, "r") as jsonfile:
        for row in jsonfile:
            a = json.loads(row)
            file_paths_queue.put(a["name"])

    total = file_paths_queue.qsize()
    # Create dynamic pool managers
    mood_manager = DynamicPoolManager(
        lambda q, rq: run_async(
            analysis_worker,
            "mood",
            args.mood_model,
            args.mood_metadata,
            q,
            rq,
            args.idle_threshold,
        ),
        mood_queue,
        args.backlog_factor,
        max_processes=args.max_processes,
    )
    genre_manager = DynamicPoolManager(
        lambda q, rq: run_async(
            analysis_worker,
            "genre",
            args.genre_model,
            args.genre_metadata,
            q,
            rq,
            args.idle_threshold,
        ),
        genre_queue,
        args.backlog_factor,
        max_processes=args.max_processes,
    )
    chord_manager = DynamicPoolManager(
        lambda q, rq: run_async(chord_worker, q, rq, args.idle_threshold),
        chord_queue,
        args.backlog_factor,
        max_processes=args.max_processes,
    )
    midi_manager = DynamicPoolManager(
        lambda q, rq: run_async(
            midi_worker, args.instrument_map, q, rq, args.idle_threshold
        ),
        midi_queue,
        args.backlog_factor,
        max_processes=args.max_processes,
    )

    all_managers = [mood_manager, genre_manager, midi_manager, chord_manager]

    # Dictionary to store partial results
    results = {}
    ct = 0
    failed_count = 0
    sofar_count = 0
    fail_timeout = 0
    last_time = time.time()

    # TODO goon
    # TODO better logging with logger
    try:
        with (
            open(args.output_file, "w") as out_json,
            open(args.failed_file, "w") as failed_json,
        ):
            while ct + failed_count < total:  # Main control loop
                # monitor logic here maybe
                if time.time() - last_time > args.monitor_interval:
                    last_time = time.time()
                    for manager in all_managers:
                        manager.monitor_and_adjust(results_queue)
                    render_processes = [p for p in render_processes if p.is_alive()]
                    while sum(p.is_alive() for p in render_processes) < args.processes:
                        spin_up_render_worker(render_processes, render_args)

                try:
                    while True:  # Process all available failed renders
                        failed_file = failed_renders_queue.get_nowait()
                        failed_json.write(json.dumps({"name": failed_file}) + "\n")
                        failed_json.flush()
                        if failed_file in results:
                            del failed_file
                        failed_count += 1
                except Empty:
                    pass  # No more failed renders to process

                # Collect and aggregate results
                try:
                    file_path, result_type, result = results_queue.get(
                        timeout=GLOBAL_TIMEOUT
                    )
                    if file_path not in results:
                        results[file_path] = {}
                    results[file_path][result_type] = result

                    # Check if we have all results for this file
                    if len(results[file_path]) == 4:  # mood, genre, midi, chords
                        rst = results[file_path]
                        audio_file = rst["chords"]["audio_file"]
                        os.remove(audio_file)
                        del rst["chords"]["audio_file"]
                        fmt = (
                            {"name": file_path}
                            | rst["genre"]
                            | rst["mood"]
                            | rst["midi"]
                            | rst["chords"]
                        )
                        out_json.write(json.dumps(fmt) + "\n")
                        out_json.flush()
                        del results[file_path]
                        ct += 1
                    sofar_count = 0
                except Empty:
                    sofar_count += 1
                    if sofar_count > 10:
                        if total - (failed_count + ct) < 10:
                            print("Approximately done, exiting")
                            sys.exit()
                        # debugger
                        print(
                            f"Haven't gotten anything in 10 secs as of {time.strftime('%Y-%m-%d %H:%M:%S')}. Is shutdown set? {shutdown_event.is_set()}"
                        )
                        print(
                            f"Number of alive render workers: {sum(p.is_alive() for p in render_processes)}/{len(render_processes)}"
                        )
                        print(f"Length of results dict: {len(results.items())}")
                        print(
                            f"Manager backlogs: {mood_manager.input_queue.qsize()}, {genre_manager.input_queue.qsize()}, {midi_manager.input_queue.qsize()}, {chord_manager.input_queue.qsize()}"
                        )
                        print(
                            f"Number of alive mood workers: {sum(p.is_alive() for p in mood_manager.processes)}/{len(mood_manager.processes)}"
                        )
                        print(
                            f"Number of alive genre workers: {sum(p.is_alive() for p in genre_manager.processes)}/{len(genre_manager.processes)}"
                        )
                        print(
                            f"Number of alive midi workers: {sum(p.is_alive() for p in midi_manager.processes)}/{len(midi_manager.processes)}"
                        )
                        print(
                            f"Number of alive chord workers: {sum(p.is_alive() for p in chord_manager.processes)}/{len(chord_manager.processes)}"
                        )
                        sofar_count = 0
                        fail_timeout += 1
                        if fail_timeout > 12:
                            print("Haven't gotten anything in 120 seconds, exiting")
                            sys.exit()
                    continue
    finally:
        print(f"{failed_count} files failed of {total}.")
        print("Cleaning up processes...")
        shutdown_event.set()

        for manager in all_managers:
            cleanup_processes(manager.processes)
        cleanup_processes(render_processes)

        queues = [
            file_paths_queue,
            mood_queue,
            genre_queue,
            midi_queue,
            chord_queue,
            results_queue,
        ]
        for queue in queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    continue
        print("Cleanup complete")


if __name__ == "__main__":
    main()
