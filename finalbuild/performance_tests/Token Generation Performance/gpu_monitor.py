"""
GPU monitoring script for testing generative agent performance
"""
import subprocess
import time
import statistics
import sys

def monitor_gpu(duration=30):
    """Monitor GPU usage for specified duration"""
    utilizations = []
    memories = []
    
    print(f"Monitoring GPU for {duration} seconds...")
    print("Press Ctrl+C to stop early\n")
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,name',
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                line = result.stdout.strip()
                parts = line.split(', ')
                util = int(parts[0])
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                gpu_name = parts[3] if len(parts) > 3 else "Unknown"
                
                utilizations.append(util)
                memories.append(mem_used)
                
                mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                print(f"[{gpu_name}] GPU: {util:3}% | Memory: {mem_used:5}MB / {mem_total}MB ({mem_percent:.1f}%)", end='\r')
            else:
                print("Error: nvidia-smi not found or GPU not available")
                return None
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    if utilizations:
        print(f"\n\n{'='*50}")
        print("GPU MONITORING RESULTS")
        print(f"{'='*50}")
        print(f"GPU Utilization:")
        print(f"  Min: {min(utilizations)}%")
        print(f"  Max: {max(utilizations)}%")
        print(f"  Average: {statistics.mean(utilizations):.1f}%")
        if len(utilizations) > 1:
            print(f"  StdDev: {statistics.stdev(utilizations):.1f}%")
        
        print(f"\nMemory Usage:")
        print(f"  Min: {min(memories)}MB")
        print(f"  Max: {max(memories)}MB")
        print(f"  Average: {statistics.mean(memories):.1f}MB")
        
        print(f"\nSamples collected: {len(utilizations)}")
        print(f"{'='*50}")
    
    return utilizations

def quick_test():
    """Quick 10-second GPU test"""
    print("Starting quick GPU test...")
    print("Make sure your generative agent server is running!\n")
    monitor_gpu(10)

def full_test():
    """Full 30-second GPU test"""
    print("Starting full GPU test...")
    print("Make sure your generative agent server is running!\n")
    monitor_gpu(30)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_test()
        elif sys.argv[1] == "full":
            full_test()
        else:
            try:
                duration = int(sys.argv[1])
                monitor_gpu(duration)
            except ValueError:
                print("Usage: python gpu_monitor.py [quick|full|<seconds>]")
    else:
        # Default 30 second test
        full_test()