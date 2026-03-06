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
    
    def _get_inner_metadata(self, doc: dict) -> dict:
        """Helper for the metadata -> metadata path."""
        return doc.get('metadata', {}).get('metadata', {})

    def _get_document_section(self, doc: dict) -> dict:
        """Helper for the metadata -> document path."""
        return doc.get('metadata', {}).get('document', {})

    def get_id(self, doc) -> str:
        meta = self._get_inner_metadata(doc)
        return meta.get('doga_id') or f"DOG-{meta.get('doga_date')}"

    def get_content(self, doc) -> str:
        # Path: metadata -> document -> content
        return self._get_document_section(doc).get('content', '')

    def get_title(self, doc) -> str:
        # Path: metadata -> document -> title
        return self._get_document_section(doc).get('title', '')

    def get_paragraph_position(self, doc: dict) -> int:
        return doc.get('metadata', {}).get('relative_chunk_id',-1)

    def get_source_id(self, doc) -> str:
        # Path: metadata -> metadata -> file_id
        return self._get_inner_metadata(doc).get('file_id', '')

    def get_score(self, doc) -> float:
        score = doc.get('score')
        return float(score) if score is not None else 0.0