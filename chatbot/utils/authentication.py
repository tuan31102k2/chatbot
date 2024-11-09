# utils/authentication.py
from pymongo import MongoClient
import hashlib
from config import MONGO_URI

# Kết nối đến MongoDB
client = MongoClient(MONGO_URI)
db = client["mydatabase"]
users_collection = db["users"]

# Hàm mã hóa mật khẩu
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Kiểm tra thông tin người dùng
def verify_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and user["password"] == hash_password(password):
        return True
    return False

# Đăng ký người dùng mới
def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return False  # Người dùng đã tồn tại
    hashed_pw = hash_password(password)
    users_collection.insert_one({"username": username, "password": hashed_pw})
    return True
