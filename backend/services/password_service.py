"""
Password Service for GRETA PAI
Implements password policy enforcement and validation
"""

import re
import hashlib
from typing import List, Tuple
from config import get_settings
import logging

logger = logging.getLogger(__name__)


# Common passwords list (top 100 most common passwords)
COMMON_PASSWORDS = {
    "password", "123456", "123456789", "12345678", "12345", "1234567", "password1",
    "123123", "1234567890", "000000", "abc123", "qwerty", "iloveyou", "monkey",
    "dragon", "111111", "letmein", "baseball", "sunshine", "master", "shadow",
    "ashley", "bailey", "passw0rd", "football", "jesus", "michael", "ninja",
    "mustang", "password123", "admin", "welcome", "login", "qwertyuiop", "solo",
    "starwars", "121212", "flower", "hottie", "loveme", "zaq1zaq1", "password1!",
    "qazwsx", "trustno1", "ranger", "buster", "thomas", "robert", "lauren",
    "1q2w3e4r", "5t6y7u8i", "1qaz2wsx", "qwerty123", "123qwe", "zxcvbnm",
    "asdfghjkl", "qwertyui", "1234qwer", "123321", "654321", "666666",
    "987654321", "123123123", "qwerty1", "qwerty12", "qwerty123456", "admin123",
    "root", "toor", "pass", "test", "guest", "info", "adm", "mysql", "user",
    "administrator", "oracle", "ftp", "pi", "puppet", "ansible", "ec2-user",
    "vagrant", "azureuser", "admin1", "admin12", "admin123", "root123", "pass123",
    "test123", "demo", "demo123", "changeme", "changeme123", "default", "password!",
}


class PasswordService:
    """Service for password validation and policy enforcement"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against policy
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check minimum length
        if len(password) < self.settings.password_min_length:
            errors.append(f"Password must be at least {self.settings.password_min_length} characters long")
        
        # Check for uppercase letters
        if self.settings.password_require_uppercase:
            if not re.search(r'[A-Z]', password):
                errors.append("Password must contain at least one uppercase letter")
        
        # Check for lowercase letters
        if self.settings.password_require_lowercase:
            if not re.search(r'[a-z]', password):
                errors.append("Password must contain at least one lowercase letter")
        
        # Check for numbers
        if self.settings.password_require_numbers:
            if not re.search(r'\d', password):
                errors.append("Password must contain at least one number")
        
        # Check for special characters
        if self.settings.password_require_special:
            if not re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', password):
                errors.append("Password must contain at least one special character (!@#$%^&*()_+-=[]{}...)")
        
        # Check against common passwords
        if self.settings.password_check_common:
            if password.lower() in COMMON_PASSWORDS:
                errors.append("Password is too common. Please choose a more unique password")
            
            # Check if password is a simple variation of common passwords
            password_lower = password.lower()
            for common in COMMON_PASSWORDS:
                if common in password_lower or password_lower in common:
                    errors.append("Password is too similar to a common password")
                    break
        
        # Check for sequential characters
        if self._has_sequential_chars(password):
            errors.append("Password contains too many sequential characters (e.g., 'abc', '123')")
        
        # Check for repeated characters
        if self._has_repeated_chars(password):
            errors.append("Password contains too many repeated characters (e.g., 'aaa', '111')")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _has_sequential_chars(self, password: str, max_sequential: int = 3) -> bool:
        """Check if password has too many sequential characters"""
        password_lower = password.lower()
        
        # Check for sequential letters (abc, xyz, etc.)
        for i in range(len(password_lower) - max_sequential + 1):
            chars = password_lower[i:i + max_sequential]
            if all(ord(chars[j]) == ord(chars[j-1]) + 1 for j in range(1, len(chars))):
                return True
            if all(ord(chars[j]) == ord(chars[j-1]) - 1 for j in range(1, len(chars))):
                return True
        
        # Check for sequential numbers (123, 987, etc.)
        for i in range(len(password) - max_sequential + 1):
            chars = password[i:i + max_sequential]
            if chars.isdigit():
                if all(int(chars[j]) == int(chars[j-1]) + 1 for j in range(1, len(chars))):
                    return True
                if all(int(chars[j]) == int(chars[j-1]) - 1 for j in range(1, len(chars))):
                    return True
        
        return False
    
    def _has_repeated_chars(self, password: str, max_repeated: int = 3) -> bool:
        """Check if password has too many repeated characters"""
        for i in range(len(password) - max_repeated + 1):
            chars = password[i:i + max_repeated]
            if len(set(chars)) == 1:  # All characters are the same
                return True
        
        return False
    
    def get_password_strength(self, password: str) -> dict:
        """
        Calculate password strength score
        
        Args:
            password: Password to analyze
            
        Returns:
            Dictionary with strength score and feedback
        """
        score = 0
        feedback = []
        
        # Length score (0-30 points)
        length = len(password)
        if length >= 16:
            score += 30
            feedback.append("✓ Excellent length")
        elif length >= 12:
            score += 20
            feedback.append("✓ Good length")
        elif length >= 8:
            score += 10
            feedback.append("○ Acceptable length")
        else:
            feedback.append("✗ Too short")
        
        # Character variety score (0-40 points)
        has_lower = bool(re.search(r'[a-z]', password))
        has_upper = bool(re.search(r'[A-Z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', password))
        
        variety_count = sum([has_lower, has_upper, has_digit, has_special])
        if variety_count == 4:
            score += 40
            feedback.append("✓ Excellent character variety")
        elif variety_count == 3:
            score += 30
            feedback.append("✓ Good character variety")
        elif variety_count == 2:
            score += 20
            feedback.append("○ Acceptable character variety")
        else:
            score += 10
            feedback.append("✗ Poor character variety")
        
        # Uniqueness score (0-30 points)
        if password.lower() not in COMMON_PASSWORDS:
            score += 15
            feedback.append("✓ Not a common password")
        else:
            feedback.append("✗ Common password detected")
        
        if not self._has_sequential_chars(password):
            score += 10
            feedback.append("✓ No sequential characters")
        else:
            feedback.append("✗ Contains sequential characters")
        
        if not self._has_repeated_chars(password):
            score += 5
            feedback.append("✓ No repeated characters")
        else:
            feedback.append("✗ Contains repeated characters")
        
        # Determine strength level
        if score >= 80:
            strength = "Very Strong"
            color = "green"
        elif score >= 60:
            strength = "Strong"
            color = "blue"
        elif score >= 40:
            strength = "Moderate"
            color = "yellow"
        elif score >= 20:
            strength = "Weak"
            color = "orange"
        else:
            strength = "Very Weak"
            color = "red"
        
        return {
            "score": score,
            "strength": strength,
            "color": color,
            "feedback": feedback,
            "max_score": 100
        }
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256
        Note: In production, use bcrypt or argon2 instead
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        # Add salt from settings
        salt = self.settings.encryption_key or "default-salt-change-in-production"
        salted_password = f"{salt}{password}{salt}"
        return hashlib.sha256(salted_password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Plain text password
            hashed_password: Hashed password to compare against
            
        Returns:
            True if password matches, False otherwise
        """
        return self.hash_password(password) == hashed_password


# Global instance
_password_service = None


def get_password_service() -> PasswordService:
    """Get password service instance"""
    global _password_service
    if _password_service is None:
        _password_service = PasswordService()
    return _password_service

