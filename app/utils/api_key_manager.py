import os
import threading
from datetime import datetime, timedelta


class APIKeyManager:
    """Rotates through multiple API keys to avoid hitting rate limits"""
    
    def __init__(self, api_keys, cooldown_minutes=5):
        if not api_keys:
            raise ValueError("Need at least one API key")
        
        self.keys = api_keys
        self.current = 0
        self.lock = threading.Lock()
        self.cooldown_time = cooldown_minutes
        
        # Track which keys have failed and when they failed
        self.failed = {}
        
        # Track usage stats
        self.requests = 0
        self.usage = {key: 0 for key in api_keys}
    
    def get_next_key(self):
        """Get the next available key, skipping any that are cooling down"""
        with self.lock:
            self.requests += 1
            
            # Clean up any keys that have finished their cooldown
            self._check_cooldowns()
            
            # Find a key that's not in cooldown
            tries = 0
            while tries < len(self.keys):
                key = self.keys[self.current]
                
                # Move to the next key for next time
                self.current = (self.current + 1) % len(self.keys)
                tries += 1
                
                # Skip keys that are in cooldown
                if key not in self.failed:
                    self.usage[key] += 1
                    return key
            
            # All keys are in cooldown - we're stuck
            raise RuntimeError(
                f"All {len(self.keys)} keys are cooling down. Wait a bit or add more keys."
            )
    
    def mark_failed(self, key):
        """Put a key in timeout after it fails"""
        with self.lock:
            if key in self.keys:
                self.failed[key] = datetime.now()
                print(f"⚠️  Key ...{key[-8:]} failed, cooling down for {self.cooldown_time} min")
    
    def mark_success(self, key):
        """Remove a key from timeout if it works again"""
        with self.lock:
            if key in self.failed:
                del self.failed[key]
                print(f"✓ Key ...{key[-8:]} is back online")
    
    def _check_cooldowns(self):
        """Check if any keys have finished their cooldown period"""
        now = datetime.now()
        timeout = timedelta(minutes=self.cooldown_time)
        
        ready = []
        for key, fail_time in self.failed.items():
            if now - fail_time > timeout:
                ready.append(key)
        
        for key in ready:
            del self.failed[key]
            print(f"✓ Key ...{key[-8:]} cooldown finished")
    
    def get_stats(self):
        """Get stats about how the keys are being used"""
        with self.lock:
            working = len(self.keys) - len(self.failed)
            
            return {
                'total_keys': len(self.keys),
                'available_keys': working,
                'keys_in_cooldown': len(self.failed),
                'total_requests': self.requests,
                'key_usage': {
                    f"key_...{key[-8:]}": count 
                    for key, count in self.usage.items()
                },
                'cooldown_minutes': self.cooldown_time
            }
    
    @classmethod
    def from_env(cls, env_var="GOOGLE_API_KEY", cooldown_minutes=5):
        """Load keys from environment variable (supports comma-separated list)"""
        key_string = os.getenv(env_var)
        
        if not key_string:
            raise ValueError(f"{env_var} not found in environment")
        
        # Split by comma and clean up whitespace
        keys = [k.strip() for k in key_string.split(',') if k.strip()]
        
        if not keys:
            raise ValueError(f"No valid keys in {env_var}")
        
        print(f"🔑 Loaded {len(keys)} API key(s)")
        
        return cls(keys, cooldown_minutes)
