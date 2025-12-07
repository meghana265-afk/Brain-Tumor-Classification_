#!/usr/bin/env bash
# Create dataset folder scaffolding for Training and Testing
set -e
mkdir -p Training/glioma Training/meningioma Training/notumor Training/pituitary
mkdir -p Testing/glioma Testing/meningioma Testing/notumor Testing/pituitary
echo "Created Training/ and Testing/ class folders (no files added)."
