# assignment2_fixed.py
# Fixed Round-Robin multitasking simulation (no caas_jupyter_tools dependency)

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import os

# Optional: pretty table output; not required
try:
    from tabulate import tabulate
    HAVE_TABULATE = True
except ImportError:
    HAVE_TABULATE = False

@dataclass
class Process:
    pid: int
    name: str
    arrival: int
    cpu_bursts: List[int]
    io_bursts: List[int] = field(default_factory=list)
    burst_idx: int = 0
    remaining_cpu: Optional[int] = None
    remaining_io: Optional[int] = None
    state: str = "new"
    finish_time: Optional[int] = None
    waiting_time: int = 0
    last_ready_enter: Optional[int] = None

    def start_next_cpu(self):
        if self.burst_idx < len(self.cpu_bursts):
            self.remaining_cpu = self.cpu_bursts[self.burst_idx]
            return True
        return False

    def start_io(self):
        if self.burst_idx < len(self.io_bursts):
            self.remaining_io = self.io_bursts[self.burst_idx]
            return True
        return False

    def is_done(self):
        return self.burst_idx >= len(self.cpu_bursts)

class RoundRobinSimulator:
    def __init__(self, processes: List[Process], quantum: int):
        self.time = 0
        self.quantum = quantum
        self.processes = {p.pid: p for p in processes}
        self.ready_queue: List[int] = []
        self.waiting_list: List[int] = []
        self.running: Optional[int] = None
        self.quantum_remaining: int = 0
        self.gantt: List[Optional[int]] = []
        self.context_switches = 0
        self.cpu_time = 0

        for p in self.processes.values():
            p.state = "new"

    def arrive_new(self):
        for p in self.processes.values():
            if p.arrival == self.time and p.state == "new":
                p.state = "ready"
                p.last_ready_enter = self.time
                p.start_next_cpu()
                self.ready_queue.append(p.pid)

    def step(self):
        self.arrive_new()

        for pid in list(self.waiting_list):
            p = self.processes[pid]
            p.remaining_io -= 1
            if p.remaining_io <= 0:
                p.burst_idx += 1
                if p.start_next_cpu():
                    p.state = "ready"
                    p.last_ready_enter = self.time + 1
                    self.ready_queue.append(pid)
                else:
                    p.state = "terminated"
                    p.finish_time = self.time + 1
                self.waiting_list.remove(pid)

        if self.running is None:
            if self.ready_queue:
                next_pid = self.ready_queue.pop(0)
                self.running = next_pid
                self.quantum_remaining = self.quantum
                p = self.processes[next_pid]
                p.state = "running"
                if p.last_ready_enter is not None:
                    p.waiting_time += (self.time - p.last_ready_enter)
                    p.last_ready_enter = None
                self.context_switches += 1

        if self.running is not None:
            p = self.processes[self.running]
            if p.remaining_cpu is None:
                p.start_next_cpu()
            p.remaining_cpu -= 1
            self.quantum_remaining -= 1
            self.cpu_time += 1
            self.gantt.append(p.pid)

            if p.remaining_cpu <= 0:
                if p.start_io():
                    p.state = "waiting"
                    self.waiting_list.append(p.pid)
                else:
                    p.burst_idx += 1
                    if p.is_done():
                        p.state = "terminated"
                        p.finish_time = self.time + 1
                    else:
                        p.start_next_cpu()
                        p.state = "ready"
                        p.last_ready_enter = self.time + 1
                        self.ready_queue.append(p.pid)
                self.running = None
            elif self.quantum_remaining <= 0:
                p.state = "ready"
                p.last_ready_enter = self.time + 1
                self.ready_queue.append(p.pid)
                self.running = None
        else:
            self.gantt.append(None)

        self.time += 1

    def run_until_done(self, max_time=10000):
        while True:
            if all(p.state == "terminated" for p in self.processes.values()):
                break
            if self.time >= max_time:
                print("Reached max_time cutoff")
                break
            self.step()

    def stats(self):
        total_time = self.time
        records = []
        for p in self.processes.values():
            tat = (p.finish_time - p.arrival) if p.finish_time is not None else None
            records.append({
                "PID": p.pid,
                "Name": p.name,
                "Arrival": p.arrival,
                "Finish": p.finish_time,
                "Turnaround": tat,
                "Waiting": p.waiting_time,
                "CPU bursts": p.cpu_bursts,
                "IO bursts": p.io_bursts
            })
        df = pd.DataFrame(records).sort_values("PID").reset_index(drop=True)
        util = (self.cpu_time / total_time) * 100 if total_time>0 else 0
        meta = {
            "Total time": total_time,
            "CPU time": self.cpu_time,
            "CPU utilization (%)": util,
            "Context switches": self.context_switches,
            "Time quantum": self.quantum,
        }
        return df, meta

    def gantt_intervals(self):
        intervals = []
        if not self.gantt:
            return intervals
        cur_pid = self.gantt[0]
        start = 0
        for t, pid in enumerate(self.gantt[1:], start=1):
            if pid != cur_pid:
                intervals.append((cur_pid, start, t))
                cur_pid = pid
                start = t
        intervals.append((cur_pid, start, len(self.gantt)))
        return intervals

def main():
    process_list = [
        Process(pid=1, name="P1", arrival=0, cpu_bursts=[5, 4], io_bursts=[3]),
        Process(pid=2, name="P2", arrival=0, cpu_bursts=[3, 2], io_bursts=[4]),
        Process(pid=3, name="P3", arrival=2, cpu_bursts=[8]),
        Process(pid=4, name="P4", arrival=1, cpu_bursts=[6, 3], io_bursts=[2]),
    ]

    quantum = 2
    sim = RoundRobinSimulator(process_list, quantum=quantum)
    sim.run_until_done(max_time=500)

    df, meta = sim.stats()

    # print table nicely if tabulate available, else fallback to pandas print
    if HAVE_TABULATE:
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    else:
        print(df.to_string(index=False))

    print("\nSummary metrics:")
    for k,v in meta.items():
        print(f" - {k}: {v}")

    intervals = sim.gantt_intervals()
    print("\nGantt intervals (pid, start, end):")
    print(intervals)

    # Save CSVs
    out_intervals = pd.DataFrame([{"pid": ("idle" if pid is None else pid), "start": start, "end": end} for pid,start,end in intervals])
    out_intervals.to_csv(os.path.join(os.getcwd(), "gantt_intervals.csv"), index=False)
    df.to_csv(os.path.join(os.getcwd(), "process_stats.csv"), index=False)
    print(f"\nSaved CSVs to: {os.path.join(os.getcwd(), 'gantt_intervals.csv')} and process_stats.csv")

    # draw and save Gantt chart as PNG
    fig, ax = plt.subplots(figsize=(10, 3))
    y = 0.5
    for pid, start, end in intervals:
        label = f"P{pid}" if pid is not None else "idle"
        ax.broken_barh([(start, end-start)], (y-0.4, 0.8), label=label)
        ax.text((start+end)/2, y, label, va='center', ha='center', fontsize=8)
    ax.set_ylim(0,1)
    ax.set_xlim(0, sim.time)
    ax.set_yticks([])
    ax.set_xlabel("Time (time units)")
    ax.set_title(f"Round-Robin Gantt Chart (quantum={quantum})")
    out_png = os.path.join(os.getcwd(), "gantt_chart.png")
    plt.savefig(out_png, bbox_inches='tight')
    print(f"Saved Gantt chart image to: {out_png}")

if __name__ == "__main__":
    main()
