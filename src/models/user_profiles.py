"""
This module manages user profiles and preferences.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from utils.database import Database

class UserProfile:
    """Manages user profiles and preferences"""
    
    def __init__(self, db: Database):
        self.db = db
        self.cache = {}
        self.last_update = {}

    def get_profile(self, user_id: int) -> Dict[str, Any]:
        """
        Get a user's profile data.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            User profile data
        """
        # Check cache first
        if user_id in self.cache:
            return self.cache[user_id]
            
        # Query database
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        
        if result:
            profile = {
                "user_id": result[0],
                "preferences": json.loads(result[1]) if result[1] else {},
                "settings": json.loads(result[2]) if result[2] else {},
                "created_at": result[3],
                "updated_at": result[4]
            }
            
            # Update cache
            self.cache[user_id] = profile
            self.last_update[user_id] = datetime.now()
            
            return profile
        else:
            return self._create_default_profile(user_id)

    def update_profile(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update a user's profile data.
        
        Args:
            user_id: The ID of the user
            updates: Dictionary of updates to apply
            
        Returns:
            Success status
        """
        try:
            current = self.get_profile(user_id)
            
            # Update preferences
            if "preferences" in updates:
                current["preferences"].update(updates["preferences"])
                
            # Update settings
            if "settings" in updates:
                current["settings"].update(updates["settings"])
                
            # Save to database
            cursor = self.db.connection.cursor()
            cursor.execute(
                """
                UPDATE users 
                SET preferences = ?, settings = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(current["preferences"]),
                    json.dumps(current["settings"]),
                    datetime.now().isoformat(),
                    user_id
                )
            )
            self.db.connection.commit()
            
            # Update cache
            self.cache[user_id] = current
            self.last_update[user_id] = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"Error updating profile: {e}")
            return False

    def _create_default_profile(self, user_id: int) -> Dict[str, Any]:
        """
        Create a default profile for a new user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Default profile data
        """
        now = datetime.now().isoformat()
        profile = {
            "user_id": user_id,
            "preferences": {
                "language": "en",
                "theme": "light",
                "notifications": True
            },
            "settings": {
                "privacy_level": "standard",
                "auto_save": True,
                "sync_enabled": True
            },
            "created_at": now,
            "updated_at": now
        }
        
        # Save to database
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO users (id, preferences, settings, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                json.dumps(profile["preferences"]),
                json.dumps(profile["settings"]),
                profile["created_at"],
                profile["updated_at"]
            )
        )
        self.db.connection.commit()
        
        # Update cache
        self.cache[user_id] = profile
        self.last_update[user_id] = datetime.now()
        
        return profile

    def get_preference(self, user_id: int, key: str) -> Any:
        """
        Get a specific user preference.
        
        Args:
            user_id: The ID of the user
            key: The preference key to retrieve
            
        Returns:
            The preference value if found, None otherwise
        """
        profile = self.get_profile(user_id)
        return profile["preferences"].get(key)

    def get_setting(self, user_id: int, key: str) -> Any:
        """
        Get a specific user setting.
        
        Args:
            user_id: The ID of the user
            key: The setting key to retrieve
            
        Returns:
            The setting value if found, None otherwise
        """
        profile = self.get_profile(user_id)
        return profile["settings"].get(key)

    def clear_cache(self, user_id: Optional[int] = None):
        """
        Clear the profile cache.
        
        Args:
            user_id: Optional specific user ID to clear
        """
        if user_id is not None:
            self.cache.pop(user_id, None)
            self.last_update.pop(user_id, None)
        else:
            self.cache = {}
            self.last_update = {}