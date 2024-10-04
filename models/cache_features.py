import os
import json
import ast
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Ensure that the sentence-transformers library is installed
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install the 'sentence-transformers' library using 'pip install sentence-transformers'.")

class FeatureCache:
    def __init__(self, cache_file='feature_cache.json', model_name='all-MiniLM-L6-v2'):
        self.cache_file = cache_file
        self.cache = []
        try:
            self.model = SentenceTransformer(model_name)  # Load the pre-trained model for embeddings
        except Exception as e:
            raise Exception(f"Error loading model '{model_name}': {e}")
        if os.path.exists(self.cache_file):
            try:
                self.load_cache()
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = []
        else:
            self.cache = []
        
        # Precompute embeddings for cached features
        for feature in self.cache:
            if 'embedding' not in feature or not feature['embedding']:
                try:
                    self._compute_feature_embedding(feature)
                except Exception as e:
                    print(f"Error computing embedding for feature '{feature.get('feature_name', 'unknown')}': {e}")
    
    def _compute_feature_embedding(self, feature_info: Dict[str, Any]):
        """
        Computes embeddings for base features and descriptions.
        """
        base_features = feature_info.get('base_features', [])
        description = feature_info.get('description', '')
        if not isinstance(base_features, list):
            base_features = []
        if not isinstance(description, str):
            description = ''
        # Combine base features and description into a single text
        text = ' '.join(map(str, base_features)) + ' ' + description
        if not text.strip():
            print(f"Warning: Empty text for embedding computation in feature '{feature_info.get('feature_name', 'unknown')}'")
            feature_info['embedding'] = []
            return
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            feature_info['embedding'] = embedding.tolist()  # Store embedding as list for JSON serialization
        except Exception as e:
            print(f"Error computing embedding for feature '{feature_info.get('feature_name', 'unknown')}': {e}")
            feature_info['embedding'] = []
    
    def add_feature(self, feature_info: Dict[str, Any]):
        """
        Adds a feature to the cache and computes its embedding.
        """
        if 'feature_name' not in feature_info:
            print("Error: 'feature_name' is missing in feature_info.")
            return
        try:
            # Compute embedding
            self._compute_feature_embedding(feature_info)
            # Check if the feature already exists in the cache
            feature_name = feature_info['feature_name']
            existing_feature = next((f for f in self.cache if f.get('feature_name') == feature_name), None)
            if existing_feature:
                # Update existing feature
                existing_feature.update(feature_info)
            else:
                self.cache.append(feature_info)
        except Exception as e:
            print(f"Error adding feature '{feature_info.get('feature_name', 'unknown')}': {e}")
    
    def update_feature_importance(self, feature_importances: Dict[str, float]):
        """
        Updates the importance scores of features in the cache.
        """
        for feature in self.cache:
            feature_name = feature.get('feature_name')
            if not feature_name:
                print("Warning: 'feature_name' missing in a feature during importance update.")
                continue
            if feature_name in feature_importances:
                feature['importance_score'] = feature_importances[feature_name]
    
    def save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        except Exception as e:
            print(f"Error loading cache from file: {e}")
            self.cache = []
    
    def retrieve_features(self, new_base_features: List[str], top_n=10):
        """
        Retrieve features from the cache whose base features are most similar to the new dataset's base features.
        Returns a list of feature_info dicts.
        """
        if not isinstance(new_base_features, list) or not new_base_features:
            print("Error: 'new_base_features' should be a non-empty list of strings.")
            return []
        # Compute embedding for new dataset's base features
        new_text = ' '.join(map(str, new_base_features))
        if not new_text.strip():
            print("Error: 'new_base_features' resulted in empty text for embedding.")
            return []
        try:
            new_embedding = self.model.encode(new_text, convert_to_numpy=True)
        except Exception as e:
            print(f"Error computing embedding for new base features: {e}")
            return []
        
        features_with_scores = []
        for feature in self.cache:
            cached_embedding_list = feature.get('embedding', [])
            if not cached_embedding_list:
                continue  # Skip if no embedding is available
            cached_embedding = np.array(cached_embedding_list)
            if cached_embedding.size == 0:
                continue
            # Compute cosine similarity
            try:
                norm_new = np.linalg.norm(new_embedding)
                norm_cached = np.linalg.norm(cached_embedding)
                if norm_new == 0 or norm_cached == 0:
                    similarity = 0
                else:
                    similarity = np.dot(new_embedding, cached_embedding) / (norm_new * norm_cached)
            except Exception as e:
                print(f"Error computing similarity with feature '{feature.get('feature_name', 'unknown')}': {e}")
                similarity = 0
            importance_score = feature.get('importance_score', 0)
            total_score = similarity * importance_score
            features_with_scores.append((total_score, feature))
        
        # Sort features by total_score
        features_with_scores.sort(key=lambda x: x[0], reverse=True)
        top_features = [feature for score, feature in features_with_scores if score > 0][:top_n]
        return top_features
    
    def apply_features_to_dataset(self, dataset: pd.DataFrame):
        """
        Applies features from the cache to the given dataset, if possible.
        Returns the dataset with new features added.
        """
        if not isinstance(dataset, pd.DataFrame):
            print("Error: 'dataset' should be a pandas DataFrame.")
            return dataset
        new_base_features = list(dataset.columns)
        applicable_features = []
        for feature in self.cache:
            base_features = feature.get('base_features', [])
            if not isinstance(base_features, list):
                continue
            # Check if all base features are present in the new dataset
            if all(f in new_base_features for f in base_features):
                applicable_features.append(feature)
        # Now, apply the transformations
        for feature in applicable_features:
            transformations = feature.get('transformations')
            if not isinstance(transformations, str):
                print(f"Warning: 'transformations' missing or invalid for feature '{feature.get('feature_name', 'unknown')}'. Skipping.")
                continue
            # Replace 'data' with 'dataset' in the code
            code_to_exec = transformations.replace('data', 'dataset')
            try:
                exec(code_to_exec, {'dataset': dataset, 'pd': pd, 'np': np})
            except Exception as e:
                print(f"Error applying feature '{feature.get('feature_name', 'unknown')}': {e}")
        return dataset

def extract_feature_info(code_text: str) -> List[Dict[str, Any]]:
    """
    Parses the code_text and extracts information about features generated.
    Returns a list of dicts with keys:
    - 'feature_name'
    - 'description'
    - 'transformations' (code snippet)
    - 'base_features' (list of base feature names)
    """
    feature_infos = []
    if not isinstance(code_text, str) or not code_text.strip():
        print("Error: 'code_text' must be a non-empty string.")
        return feature_infos
    try:
        # Split the code_text into individual code blocks
        code_blocks = code_text.strip().split('\n\n')
        for block in code_blocks:
            if not block.strip():
                continue
            try:
                # Parse each code block separately
                tree = ast.parse(block)
            except Exception as e:
                print(f"Error parsing code block: {e}")
                continue
            # Initialize variables
            feature_name = ''
            description = ''
            transformations = block
            base_features = set()
            # Extract descriptions and feature names
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    # Handle features_description assignments
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                        subscript = node.targets[0]
                        if isinstance(subscript.value, ast.Name) and subscript.value.id == 'features_description':
                            if isinstance(subscript.slice, (ast.Index, ast.Constant)):
                                feature_name_node = subscript.slice.value if isinstance(subscript.slice, ast.Index) else subscript.slice
                                if isinstance(feature_name_node, ast.Str):
                                    feature_name = feature_name_node.s
                                elif isinstance(feature_name_node, ast.Constant) and isinstance(feature_name_node.value, str):
                                    feature_name = feature_name_node.value
                                # Get the description from the value
                                if isinstance(node.value, ast.Str):
                                    description = node.value.s
                                elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                    description = node.value.value
                    # Handle data assignments (features)
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                        subscript = node.targets[0]
                        if isinstance(subscript.value, ast.Name) and subscript.value.id == 'data':
                            if isinstance(subscript.slice, (ast.Index, ast.Constant)):
                                feature_name_node = subscript.slice.value if isinstance(subscript.slice, ast.Index) else subscript.slice
                                if isinstance(feature_name_node, ast.Str):
                                    feature_name = feature_name_node.s
                                elif isinstance(feature_name_node, ast.Constant) and isinstance(feature_name_node.value, str):
                                    feature_name = feature_name_node.value
                                # Now, find the base features used in the value
                                base_features.update(find_base_features(node.value))
            if feature_name:
                feature_info = {
                    'feature_name': feature_name,
                    'description': description,
                    'transformations': transformations,
                    'base_features': list(base_features),
                    'importance_score': 0.0  # Initialize importance score
                }
                feature_infos.append(feature_info)
            else:
                print(f"Warning: Could not extract 'feature_name' from code block:\n{block}")
        return feature_infos
    except Exception as e:
        print(f"Error parsing code: {e}")
        return []

def find_base_features(node):
    """
    Recursively finds base features in an AST node.
    Returns a set of base feature names.
    """
    base_features = set()
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == 'data':
            if isinstance(node.slice, (ast.Index, ast.Constant)):
                if isinstance(node.slice, ast.Index):
                    base_feature_node = node.slice.value
                else:  # For Python 3.9 and above
                    base_feature_node = node.slice
                if isinstance(base_feature_node, ast.Str):
                    base_feature_name = base_feature_node.s
                    base_features.add(base_feature_name)
                elif isinstance(base_feature_node, ast.Constant) and isinstance(base_feature_node.value, str):
                    base_feature_name = base_feature_node.value
                    base_features.add(base_feature_name)
    for child in ast.iter_child_nodes(node):
        base_features.update(find_base_features(child))
    return base_features

# Example usage:

# Initialize the feature cache
feature_cache = FeatureCache(cache_file='feature_cache.json')

# Assume 'llm_code' is the code generated by the LLM during feature generation
llm_code = """
#1. Feature 1.
features_description['new_feature'] = 'Complex transformation of base features: feature_A, feature_B, and feature_C'
for i in range(3):
    data['new_feature'] = np.log1p(data['feature_A'] + data['feature_B'] + i * data['feature_C'])
    data['new_feature'] = data['new_feature'].rolling(window=2).mean()

#2. Feature 2.
features_description['another_feature'] = 'Another complex transformation involving feature_D and feature_E'
data['temp_feature'] = data['feature_D'] ** 2
data['another_feature'] = data['temp_feature'] / data['feature_E']
del data['temp_feature']
"""

# Extract feature information from the LLM code
feature_infos = extract_feature_info(llm_code)
for feature_info in feature_infos:
    feature_cache.add_feature(feature_info)

# After training the model and obtaining feature importances
# Assume 'feature_importances' is a dictionary of feature names and their importance scores
feature_importances = {
    'new_feature': 0.9,
    'another_feature': 0.7
}

# Update the cache with importance scores
feature_cache.update_feature_importance(feature_importances)

# Save the cache to disk
feature_cache.save_cache()

# Later, given a new dataset
new_dataset = pd.DataFrame({
    'feature_A': np.random.rand(100),
    'feature_B': np.random.rand(100),
    'feature_C': np.random.rand(100),
    'feature_D': np.random.rand(100),
    'feature_E': np.random.rand(100)
})

new_base_features = list(new_dataset.columns)

# Retrieve relevant features from the cache
top_features = feature_cache.retrieve_features(new_base_features, top_n=10)

# Print the names of the retrieved features
print("Retrieved Features:")
for feature in top_features:
    print(f"- {feature['feature_name']} (Importance Score: {feature.get('importance_score', 0)})")

# Apply the retrieved features to the new dataset
updated_dataset = feature_cache.apply_features_to_dataset(new_dataset)

# Now 'updated_dataset' contains the new features applied from the cache
print(updated_dataset.head())
