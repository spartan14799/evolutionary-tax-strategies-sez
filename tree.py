import os

def print_tree_2_depth(root="."):
    print(root)
    
    items = sorted(os.listdir(root))
    for i, item in enumerate(items):
        path = os.path.join(root, item)
        is_last = i == len(items) - 1
        
        prefix = "└── " if is_last else "├── "
        print(prefix + item + ("/" if os.path.isdir(path) else ""))
        
        # Nivel 2 (solo si es carpeta)
        if os.path.isdir(path):
            subitems = sorted(os.listdir(path))
            for j, sub in enumerate(subitems):
                subpath = os.path.join(path, sub)
                is_sub_last = j == len(subitems) - 1
                
                sub_prefix = "    " if is_last else "│   "
                sub_connector = "└── " if is_sub_last else "├── "
                
                print(sub_prefix + sub_connector + sub + ("/" if os.path.isdir(subpath) else ""))

print_tree_2_depth()