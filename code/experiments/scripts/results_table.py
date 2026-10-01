import json, math, os, csv, sys
L=lambda p:{r["id"]:r for r in map(json.loads,open(p))}
B=L("/fsx/opd/baseline-gsm8k/samples_before.jsonl"); T=L("/fsx/opd/baseline-gsm8k/samples_teacher.jsonl")
bb=sum(v["correct"] for v in B.values())/len(B); tt=sum(v["correct"] for v in T.values())/len(T)
RUNS=[("run13","/fsx/opd/eval-run13",[25,50,75,100],"1e-5",100),
      ("run14","/fsx/opd/eval-run14",[25,50,75,100,125,150,175,200],"1e-5",200),
      ("run15","/fsx/opd/eval-run15",[25,50,75,100],"2e-5",100)]
w=csv.writer(sys.stdout)
w.writerow(["run","lr","total_steps","step","accuracy","delta_vs_base_pp","fixed","broken","z_paired","gap_closed","answer_marker","capped","mean_tokens"])
for name,d,steps,lr,total in RUNS:
    for s in steps:
        p=f"{d}/step{s}/samples_after.jsonl"
        if not os.path.exists(p): continue
        A=L(p); ids=sorted(A.keys()&B.keys()); n=len(ids); acc=sum(A[i]["correct"] for i in ids)/n
        f=sum(A[i]["correct"] and not B[i]["correct"] for i in ids); b=sum(B[i]["correct"] and not A[i]["correct"] for i in ids)
        toks=[A[i]["output_tokens"] or 0 for i in ids]
        w.writerow([name,lr,total,s,f"{acc:.4f}",f"{(acc-bb)*100:+.1f}",f,b,f"{(f-b)/math.sqrt(max(1,f+b)):+.2f}",f"{(acc-bb)/(tt-bb):.3f}",
                    f"{sum(A[i]['has_marker'] for i in ids)/n:.3f}",f"{sum(t>=1020 for t in toks)/n:.3f}",f"{sum(toks)/n:.0f}"])
