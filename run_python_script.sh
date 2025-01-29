#!/bin/bash
while true; do
    python get_all_dataframes.py
    if [ $? -eq 0 ]; then
        echo "Script completed successfully."
        break
    else
        echo "Script crashed. Restarting..."
        sleep 5  # Optional delay before restarting
    fi
done
