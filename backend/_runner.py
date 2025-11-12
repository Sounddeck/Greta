import sys, traceback, runpy, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    runpy.run_path(os.path.join(os.path.dirname(__file__), 'main.py'))
except SystemExit as e:
    # allow clean exits
    pass
except Exception:
    with open(os.path.join(os.path.dirname(__file__), 'run_err.log'), 'w') as f:
        f.write(traceback.format_exc())
    raise

