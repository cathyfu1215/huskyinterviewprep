from app import create_app

app = create_app()

if __name__ == "__main__":
    # Ensure static and templates directories exist
    import os
    if not os.path.exists('app/templates'):
        os.makedirs('app/templates')
    
    if not os.path.exists('app/static'):
        os.makedirs('app/static')
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5002) 