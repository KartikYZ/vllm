# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""RAG-aware cache management layer on top of block-based KV cache."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.cedar_utils.tokenizer_utils import get_tokenizer

logger = init_logger(__name__)

@dataclass
class ChunkInfo:
    """Information about a chunk and its associated blocks."""
    start_idx: int  # Start token index
    end_idx: int  # End token index
    block_ids: List[int]  # List of block IDs belonging to this chunk
    last_used: float  # Timestamp of last usage
    size: int  # Total size in tokens

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def key(self) -> str:
        """Unique identifier for the chunk based on its block IDs."""
        return f"block_{min(self.block_ids)}_{max(self.block_ids)}"

class RAGCacheManager:
    """Manages caching at chunk level while delegating block operations to KVCacheManager."""
    
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.chunks: Dict[str, ChunkInfo] = {}  # chunk_key -> ChunkInfo
        self.block_to_chunk: Dict[int, str] = {}  # block_id -> chunk_key
        self.total_chunks = 0
        self.total_blocks = 0
        
        # Get tokenizer from global state
        tokenizer = get_tokenizer()
        if tokenizer:
            self.boundary_token_ids = set([
                tokenizer.encode("---")[0],  # Start of chunk marker
                tokenizer.encode("\n")[0],   # Newline
            ])
        else:
            logger.warning("No tokenizer found, using empty boundary tokens")
            self.boundary_token_ids = set()

    def detect_chunks_from_tokens(self, token_ids: List[int]) -> List[ChunkInfo]:
        """Detect chunks based on token patterns instead of text."""
        chunks = []
        current_start = 0
        
        # Look for boundary patterns in tokens
        for i, token_id in enumerate(token_ids):
            if token_id in self.boundary_token_ids:
                if i > current_start:
                    # Calculate block IDs for this chunk
                    chunk_size = i - current_start
                    start_block = current_start // self.block_size
                    end_block = cdiv(i, self.block_size)
                    block_ids = list(range(start_block, end_block))
                    
                    chunks.append(ChunkInfo(
                        start_idx=current_start,
                        end_idx=i,
                        block_ids=block_ids,
                        last_used=0.0,
                        size=chunk_size
                    ))
                    
                current_start = i + 1
                
        # Handle the last chunk
        if current_start < len(token_ids):
            chunk_size = len(token_ids) - current_start
            start_block = current_start // self.block_size
            end_block = cdiv(len(token_ids), self.block_size)
            block_ids = list(range(start_block, end_block))
            
            chunks.append(ChunkInfo(
                start_idx=current_start,
                end_idx=len(token_ids),
                block_ids=block_ids,
                last_used=0.0,
                size=chunk_size
            ))
            
        return chunks

    def add_chunks(self, chunks: List[ChunkInfo]) -> None:
        """Add new chunks and their block mappings."""
        for chunk in chunks:
            chunk_key = chunk.key
            self.chunks[chunk_key] = chunk
            for block_id in chunk.block_ids:
                self.block_to_chunk[block_id] = chunk_key
            self.total_chunks += 1
            self.total_blocks += chunk.num_blocks

    def get_blocks_to_evict(self, needed_blocks: int) -> Set[int]:
        """Get blocks to evict based on chunk-level LRU."""
        if needed_blocks <= 0:
            return set()

        # Sort chunks by last_used timestamp
        sorted_chunks = sorted(
            self.chunks.values(), 
            key=lambda x: x.last_used
        )

        blocks_to_evict = set()
        for chunk in sorted_chunks:
            blocks_to_evict.update(chunk.block_ids)
            if len(blocks_to_evict) >= needed_blocks:
                break

        # Remove chunk mappings for evicted blocks
        for block_id in blocks_to_evict:
            chunk_key = self.block_to_chunk.pop(block_id, None)
            if chunk_key:
                chunk = self.chunks.pop(chunk_key, None)
                if chunk:
                    self.total_chunks -= 1
                    self.total_blocks -= chunk.num_blocks

        return blocks_to_evict

    def update_chunk_usage(self, block_ids: List[int], timestamp: float) -> None:
        """Update last_used timestamp for chunks containing the given blocks."""
        seen_chunks = set()
        for block_id in block_ids:
            chunk_key = self.block_to_chunk.get(block_id)
            if chunk_key and chunk_key not in seen_chunks:
                chunk = self.chunks.get(chunk_key)
                if chunk:
                    chunk.last_used = timestamp
                    seen_chunks.add(chunk_key)

    def get_chunk_info(self) -> dict:
        """Get current cache statistics."""
        return {
            "total_chunks": self.total_chunks,
            "total_blocks": self.total_blocks,
            "chunks": {
                chunk.key: {
                    "size": chunk.size,
                    "num_blocks": chunk.num_blocks,
                    "last_used": chunk.last_used
                }
                for chunk in self.chunks.values()
            }
        }

    def get_cached_blocks(self, chunk_key: str) -> Optional[List[int]]:
        """Get the block IDs associated with a chunk."""
        chunk = self.chunks.get(chunk_key)
        if chunk:
            return chunk.block_ids
        return None

    def remove_blocks(self, block_ids: List[int]) -> None:
        """Remove blocks and their associated chunks from the cache."""
        chunks_to_remove = set()
        
        # Find chunks containing these blocks
        for block_id in block_ids:
            chunk_key = self.block_to_chunk.pop(block_id, None)
            if chunk_key:
                chunks_to_remove.add(chunk_key)
        
        # Remove the chunks
        for chunk_key in chunks_to_remove:
            if chunk := self.chunks.pop(chunk_key, None):
                self.total_chunks -= 1
                self.total_blocks -= chunk.num_blocks 