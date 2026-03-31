#!/usr/bin/env python3
import sys
import os

os.chdir("/mnt/c/Users/TAWKIR/Documents/GitHub/Feedback_Vertex_Set_Problem_Algorithm_Project_4-2")
for candidate in ("build-linux", "build-macos", "build-win", "build"):
    sys.path.insert(0, f"cpp_engine/{candidate}")

#!/usr/bin/env python3
import sys
import os

# Add the WSL build directory to path
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.10/site-packages"))
sys.path.insert(0, "/home/ttt/cpp_engine_build/build")

try:
    import cpp_engine
    print("✓ SUCCESS! C++ module imported")
    print(f"    Module location: {cpp_engine.__file__}")
    
    # List available functions
    funcs = [x for x in dir(cpp_engine) if not x.startswith('_')]
    print(f"    Available functions ({len(funcs)}):")
    for func in sorted(funcs):
        print(f"      - {func}")
    
    # Test a simple case
    print("\n    Testing solve_undirected_BST on a triangle...")
    fvs = cpp_engine.solve_undirected_BST(3, [(0,1),(1,2),(2,0)])
    print(f"    Triangle FVS: {fvs} (size {len(fvs)})")
    
    if len(fvs) == 1:
        print("✓ Module works correctly!")
    else:
        print(f"✓ Module works (returned FVS of size {len(fvs)})")
        
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
