#!/bin/bash

CONFIG_DIR="examples/config"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./feature-eng.sh --load_config "$file"
done

echo "All configurations processed."
