#!/bin/bash

# Run tests with coverage
pytest test_flask_app.py --cov=flask_app --cov-report=term-missing 