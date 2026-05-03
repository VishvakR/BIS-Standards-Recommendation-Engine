from typing import Dict, Type
from .main import IndexBase

class IndexFactory:
    
    _registry = {}

    @classmethod
    def registry(cls, store_type: str):
        def decorator(wrapped_class: Type[IndexBase]):
            cls._registry[store_type.lower()] = wrapped_class
            return wrapped_class
        return decorator
    
    @classmethod
    def create(cls, store_type: str, **kwargs) -> IndexBase:
        index_cls = cls._registry.get(store_type.lower())
        if not index_cls:
            raise ValueError(f"Store type '{store_type}' not registered")
        try:
            return index_cls(**kwargs)
        except TypeError as e:
            raise ValueError(f"Error creating index: {e}")
        
class IndexManager:
    @staticmethod
    def create(store_type: str, **kwargs) -> IndexBase:
        return IndexFactory.create(store_type, **kwargs)
    
    
