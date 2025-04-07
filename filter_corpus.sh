#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p corpus_success

# Counter for processed files
total=0
success=0

# Count total files first
total_files=$(ls corpus/*.mlir 2>/dev/null | wc -l)
current=0

# Process each .mlir file in the corpus directory
for file in corpus/*.mlir; do
    if [ -f "$file" ]; then
        total=$((total + 1))
        current=$((current + 1))
        filename=$(basename "$file")
        
        # Calculate progress percentage
        percent=$((current * 100 / total_files))
        
        # Create progress bar
        bar=""
        for ((i=1; i<=50; i++)); do
            if [ $i -le $((percent / 2)) ]; then
                bar="${bar}#"
            else
                bar="${bar}-"
            fi
        done
        
        # Print progress bar
        printf "\r[%s] %d%% (%d/%d) - Processing: %s" "$bar" "$percent" "$current" "$total_files" "$filename"
        
        # Run mlir-opt and capture its output and exit status
        if mlir-opt "$file" > /dev/null 2>&1; then
            # If mlir-opt succeeded (exit status 0), copy the file to corpus_success
            cp "$file" "corpus_success/$filename"
            success=$((success + 1))
        else
            echo "Failed: $filename"
        fi
    fi
done

# Print newline after progress bar
echo -e "\n"

# Print summary
echo "Processing complete!"
echo "Total files processed: $total"
echo "Successful files: $success"
echo "Failed files: $((total - success))" 