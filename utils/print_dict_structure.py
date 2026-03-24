def print_structure(data, indent=0):
    """
    Recursively prints the structure of a nested dictionary or list,
    showing keys and data types without the actual data.
    """
    spacing = "    " * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{spacing}- {key}: <dict>")
                print_structure(value, indent + 1)
            elif isinstance(value, list):
                print(f"{spacing}- {key}: <list> (length: {len(value)})")
                print_structure(value, indent + 1)
            else:
                # Base case: standard variable (str, int, bool, etc.)
                print(f"{spacing}- {key}: <{type(value).__name__}>")
                
    elif isinstance(data, list):
        if data:
            print(f"{spacing}[First element structure]:")
            # Only check the first element to avoid printing 1,000 identical list item structures
            print_structure(data[0], indent + 1)
        else:
            print(f"{spacing}[Empty list]")