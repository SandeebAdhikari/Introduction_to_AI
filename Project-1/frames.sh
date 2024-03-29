#!/bin/bash
# Start the Python script in the background
python ./src/frames.py &

# Start the Jupyter notebook server in the foreground
exec jupyter notebook --ip 0.0.0.0 --port=7777 --no-browser --allow-root
