import json
import time
import copy
import lmstudio as lm

with open("test.json", "r") as f:
    ctx = lm.ChatContext.from_history(json.loads(f.read()))

template = "genre tags: {genre_tags} \n mood tags: {mood_tags}\n instruments summary tags: {ist}\n time signature tags: {tst}\n tempo tags: {tempo}\n key tags: {key}\n chords summary tags: {cst}\n duration tags: {dur}"
llm = lm.Client(api_host="phantasmagoria.local:1234").llm.get("qwen2.5-7b-instruct-1m")

start_time = time.time()
ct = 0
try:
    with open("1.json", "r") as f, open("1_captions.jsonl", "a") as fout:
        i = 0
        for line in f:
            i += 1
            if i < 141509:
                continue
            ncc = copy.deepcopy(ctx)
            lineobj = json.loads(line)

            genre = [sublist[:2] for sublist in lineobj.get("genre")]
            mood = lineobj.get("mood")
            ist = lineobj.get("mapped_instruments_summary")
            tst = lineobj.get("time_signature")
            tempo = lineobj.get("tempo")
            key =lineobj.get("key")
            cst = lineobj.get("chord_summary")
            dur = lineobj.get("duration")

            ncc.add_user_message(template.format(
                    genre_tags=genre,
                    mood_tags=mood,
                    ist=ist,
                    tst=tst,
                    tempo=tempo,
                    key=key,
                    cst=cst,
                    dur=dur
                ))

            response = llm.respond(ncc)
            json_out = {
                "location": lineobj.get("name"),
                "caption": response.content,
                "genre": genre[0],
                "genre_prob": genre[1],
                "mood": mood[0],
                "mood_prob": mood[1],
                "key": key,
                "time_signature": tst,
                "tempo": tempo[0],
                "tempo_word": tempo[1],
                "duration": dur[0],
                "duration_word": dur[1],
                "chord_summary": cst[0],
                "chord_summary_occurrence": cst[1],
                "instrument_summary": ist,
                "instrument_numbers_sorted": lineobj.get("sorted_instrurments"),
                "all_chords": [chord[0] for chord in lineobj.get("chords")],
                "all_chords_timestamps": [chord[1] for chord in lineobj.get("chords")],
                "test_set": False
            }
            fout.write(str(json_out) + "\n")
            fout.flush()
            ct += 1
finally:
    end_time = time.time()
    print(f"Average time per line: {(end_time - start_time) / ct} secs")