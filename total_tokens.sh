#!/bin/bash

# Define the file containing the numbers
FILE="total_tokens.txt"

# Initialize sum variable
sum=0

# Read through each line in the file and add the numbers to the sum
while IFS= read -r line; do
  sum=$((sum + line))
done < "$FILE"

# Print the total sum
echo "The total sum is: $sum"