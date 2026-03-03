from abc import ABC, abstractmethod

class BaseDocumentAdapter(ABC):
    @abstractmethod
    def get_id(self, doc) -> str: pass
    
    @abstractmethod
    def get_content(self, doc) -> str: pass
    
    @abstractmethod
    def get_title(self, doc) -> str: pass
    
    @abstractmethod
    def get_paragraph_position(self, doc) -> int: pass

    @abstractmethod
    def get_source_id(self, doc) -> str: pass

    @abstractmethod
    def get_score(self, doc) -> float: pass

class PressAdapter(BaseDocumentAdapter):

    def get_id(self, doc) -> str:
        metadata = doc.get('metadata', {})
        return metadata.get('id')
    
    def get_content(self, doc) -> str:
        return doc.get('content', '')

    def get_title(self, doc) -> str:
        metadata = doc.get('metadata', {})
        return metadata.get('title') or metadata.get('headline', '')

    def get_paragraph_position(self, doc) -> int:
        metadata = doc.get('metadata', {})
        return metadata.get('relative_chunk_id', -1)

    def get_source_id(self, doc) -> str:
        metadata = doc.get('metadata', {})
        return metadata.get('source_id') or f"Praza-{metadata.get('published_on')}"
    
    def get_score(self, doc) -> float:
        metadata = doc.get('metadata', {})
        return metadata.get('score', 0.0)
        

class DOGAdapter(BaseDocumentAdapter):

    def get_id(self, doc) -> str:
        pass
    
    def get_content(self, doc) -> str:
        pass

    def get_title(self, doc) -> str:
        pass

    def get_paragraph_position(self, doc) -> int:
        pass

    def get_source_id(self, doc) -> str:
        pass
    
    def get_score(self, doc) -> float:
        pass