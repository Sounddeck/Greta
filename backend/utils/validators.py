"""
Validation utilities for GRETA PAI
Provides validators for email, phone, password, and other user input
"""

import re
from typing import Optional
from pydantic import validator


def email_validator(email: str) -> bool:
    """
    Validate email address format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_email_address(email: str) -> str:
    """
    Validate and return email address
    
    Args:
        email: Email address to validate
        
    Returns:
        Validated email address
        
    Raises:
        ValueError: If email is invalid
    """
    if not email_validator(email):
        raise ValueError("Invalid email address format")
    return email.lower()


def phone_validator(phone: str) -> bool:
    """
    Validate phone number format
    Accepts various formats: +1234567890, (123) 456-7890, 123-456-7890, etc.
    
    Args:
        phone: Phone number to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)\.]', '', phone)
    
    # Check if it's a valid phone number (10-15 digits, optional + prefix)
    pattern = r'^\+?[1-9]\d{9,14}$'
    return bool(re.match(pattern, cleaned))


def validate_phone_number(phone: str) -> str:
    """
    Validate and return phone number
    
    Args:
        phone: Phone number to validate
        
    Returns:
        Validated phone number
        
    Raises:
        ValueError: If phone number is invalid
    """
    if not phone_validator(phone):
        raise ValueError("Invalid phone number format")
    return phone


def password_validator(password: str) -> bool:
    """
    Validate password strength
    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    
    Args:
        password: Password to validate
        
    Returns:
        True if valid, False otherwise
    """
    if len(password) < 8:
        return False
    
    has_upper = bool(re.search(r'[A-Z]', password))
    has_lower = bool(re.search(r'[a-z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    
    return has_upper and has_lower and has_digit and has_special


def validate_password_strength(password: str) -> str:
    """
    Validate password strength and return it
    
    Args:
        password: Password to validate
        
    Returns:
        Validated password
        
    Raises:
        ValueError: If password doesn't meet requirements
    """
    if not password_validator(password):
        raise ValueError(
            "Password must be at least 8 characters and contain: "
            "uppercase letter, lowercase letter, digit, and special character"
        )
    return password


def validate_username(username: str) -> str:
    """
    Validate username format
    Requirements:
    - 3-30 characters
    - Alphanumeric and underscores only
    - Must start with a letter
    
    Args:
        username: Username to validate
        
    Returns:
        Validated username
        
    Raises:
        ValueError: If username is invalid
    """
    if not username or len(username) < 3 or len(username) > 30:
        raise ValueError("Username must be 3-30 characters")
    
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', username):
        raise ValueError(
            "Username must start with a letter and contain only "
            "letters, numbers, and underscores"
        )
    
    return username


def url_validator(url: str) -> bool:
    """
    Validate URL format

    Args:
        url: URL to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
    return bool(re.match(pattern, url))


def validate_url(url: str) -> str:
    """
    Validate URL format and return it

    Args:
        url: URL to validate

    Returns:
        Validated URL

    Raises:
        ValueError: If URL is invalid
    """
    if not url_validator(url):
        raise ValueError("Invalid URL format")
    return url

