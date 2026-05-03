from typing import Dict, Type
from .main import Store

# A registry to manage different store classes
class StorageFactory:
    _registry: Dict[str, Type[Store]] = {}

    # Register a store class with a given name
    @classmethod
    def register_store(cls, store_type: str):
        def decorator(wrapper_class: Type[Store]):
            cls._registry[store_type.lower()] = wrapper_class
            return wrapper_class
        return decorator
    

    @classmethod
    def get_or_create(cls, store_type: str, **kwargs):
        model = cls._registry.get(store_type)
        if not model:
            raise ValueError(f"Model type '{store_type}' is not registered.")
        try:
            return model(**kwargs)
        except TypeError as e:
            raise ValueError(f"Error initializing '{store_type}' index: {e}")

class StoreManager:
    @staticmethod
    def get_or_create(store_type: str, **kwargs) -> Store:
        return StorageFactory.get_or_create(store_type, **kwargs)
    
    
