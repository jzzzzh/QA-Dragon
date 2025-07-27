from typing import Dict, List, Any, Optional
from PIL import Image

from cragmm_search.search import UnifiedSearchPipeline

class MetaBaseAgent:
    """
    BaseAgent is the abstract base class for all CRAG-MM benchmark agents.
    
    Any agent implementation for the CRAG-MM benchmark should inherit from this class
    and implement the required methods. The agent is responsible for generating responses
    to user queries, potentially using images and conversation history for context.
    
    The CRAG-MM evaluation framework evaluates agents on both single-turn and 
    multi-turn conversation tasks.
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
            Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
    """
    
    def __init__(self, search_pipeline: UnifiedSearchPipeline):
        """
        Initialize the BaseAgent.
        
        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
            Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
        """
        self.search_pipeline = search_pipeline
    
    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.
        
        Returns:
            int: The batch size, indicating how many queries should be processed together 
                 in a single batch.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate responses for a batch of queries.
        
        This is the main method called by the evaluator. It processes multiple
        queries in parallel for efficiency. For multi-turn conversations,
        the message_histories parameter contains the conversation so far.
        
        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects, one per query. 
                The evaluator will ensure that the dataset rows which have just
                image_url are populated with the associated image.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories,
                one per query. Each history is a list of message dictionaries with
                'role' and 'content' keys in the following format:
                
                - For single-turn conversations: Empty list []
                - For multi-turn conversations: List of previous message turns in the format:
                  [
                    {"role": "user", "content": "first user message"},
                    {"role": "assistant", "content": "first assistant response"},
                    {"role": "user", "content": "follow-up question"},
                    {"role": "assistant", "content": "follow-up response"},
                    ...
                  ]
                
        Returns:
            List[str]: List of generated responses, one per input query.
        """
        raise NotImplementedError("Subclasses must implement this method")
