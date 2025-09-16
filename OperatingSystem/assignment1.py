import threading
import time
import random

def task_function(task_id, iterations=5):
    """
    Simulates a task in a multitasking environment.
    Each task performs CPU work and I/O waits in a loop.
    """
    for i in range(iterations):
        # Simulate CPU-bound work: a simple calculation
        start_time = time.time()
        total = sum(range(100000))  # Quick computation to mimic CPU usage
        cpu_time = time.time() - start_time
        
        # Print CPU work status
        print(f"[{time.strftime('%H:%M:%S')}] Task {task_id} - Iteration {i+1}: CPU work done (took {cpu_time:.3f}s, sum={total})")
        
        # Simulate I/O wait (preemption point)
        io_delay = random.uniform(0.5, 1.5)  # Random sleep to vary interleaving
        print(f"[{time.strftime('%H:%M:%S')}] Task {task_id} - Starting I/O wait for {io_delay:.1f}s...")
        time.sleep(io_delay)
        print(f"[{time.strftime('%H:%M:%S')}] Task {task_id} - I/O complete")

def main():
    """
    Main function: Creates and starts multiple tasks (threads).
    Demonstrates multitasking via concurrent execution.
    """
    num_tasks = 3
    threads = []
    
    print("Starting multitasking simulation with 3 tasks...")
    print("Observe interleaved output to see context switching.\n")
    
    # Create and start threads
    for i in range(num_tasks):
        thread = threading.Thread(target=task_function, args=(chr(65 + i),))  # Task A, B, C
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("\nAll tasks completed. Simulation end.")

if __name__ == "__main__":
    main()