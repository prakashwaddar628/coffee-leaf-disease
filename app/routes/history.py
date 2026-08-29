from fastapi import APIRouter
from typing import List
from app.schemas import HistoryItem
from app.storage import storage

router = APIRouter()

@router.get("/history", response_model=List[HistoryItem])
async def get_history():
    return storage.get_history()
