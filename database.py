import pymssql
import os
from dotenv import load_dotenv
from config import DATABASE_CONFIG
import logging

# Logging ayarları
logging.basicConfig(level=logging.ERROR)  # Sadece ERROR seviyesi
logger = logging.getLogger(__name__)

# .env dosyasını yükle (eğer varsa)
load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.config = self._get_database_config()
    
    def _get_database_config(self):
        """Veritabanı konfigürasyonunu al (environment variables öncelikli)"""
        config = DATABASE_CONFIG.copy()
        
        # Environment variables'dan değerleri al
        if os.getenv('DB_SERVER'):
            config['server'] = os.getenv('DB_SERVER')
        if os.getenv('DB_NAME'):
            config['database'] = os.getenv('DB_NAME')
        if os.getenv('DB_USERNAME'):
            config['username'] = os.getenv('DB_USERNAME')
        if os.getenv('DB_PASSWORD'):
            config['password'] = os.getenv('DB_PASSWORD')
        if os.getenv('DB_PORT'):
            config['port'] = int(os.getenv('DB_PORT'))

        
        return config
    
    def connect(self):
        """Veritabanına bağlan"""
        try:
            if self.connection is None:
                logger.info(f"Connecting to database: {self.config['server']}:{self.config['port']}/{self.config['database']}")
                
                # pymssql'de trusted_connection parametresi yok, sadece SQL Authentication kullan
                self.connection = pymssql.connect(
                    server=self.config['server'],
                    port=self.config['port'],
                    database=self.config['database'],
                    user=self.config['username'],
                    password=self.config['password'],
                    timeout=self.config['timeout']
                )
                
                logger.info("Database connection successful")
            return self.connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Veritabanı bağlantısını kapat"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def test_connection(self):
        """Veritabanı bağlantısını test et"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """SQL sorgusu çalıştır ve sonuçları döndür"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # SELECT sorgusu ise sonuçları al
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [column[0] for column in cursor.description]
                cursor.close()
                return [dict(zip(columns, row)) for row in results]
            else:
                # INSERT, UPDATE, DELETE sorgusu
                conn.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                return affected_rows
                
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_image_vectors(self):
        """Veritabanından görsel vektörlerini al"""
        query = """
        SELECT 
            image_id,
            image_path,
            image_name,
            dino_vector,
            clip_vector,
            combined_vector,
            upload_date,
            metadata
        FROM image_vectors
        ORDER BY upload_date DESC
        """
        
        try:
            results = self.execute_query(query)
            logger.info(f"Retrieved {len(results)} image vectors from database")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve image vectors: {e}")
            return []
    
    def insert_image_vector(self, image_path, image_name, dino_vector, clip_vector, combined_vector, metadata=None):
        """Yeni görsel vektörünü veritabanına ekle"""
        query = """
        INSERT INTO image_vectors 
        (image_path, image_name, dino_vector, clip_vector, combined_vector, upload_date, metadata)
        VALUES (?, ?, ?, ?, ?, GETDATE(), ?)
        """
        
        try:
            result = self.execute_query(query, (
                image_path, 
                image_name, 
                dino_vector.tobytes(), 
                clip_vector.tobytes(), 
                combined_vector.tobytes(),
                metadata
            ))
            logger.info(f"Image vector inserted successfully: {image_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to insert image vector: {e}")
            raise
    
    def update_image_vector(self, image_id, dino_vector, clip_vector, combined_vector, metadata=None):
        """Mevcut görsel vektörünü güncelle"""
        query = """
        UPDATE image_vectors 
        SET dino_vector = ?, clip_vector = ?, combined_vector = ?, metadata = ?
        WHERE image_id = ?
        """
        
        try:
            result = self.execute_query(query, (
                dino_vector.tobytes(), 
                clip_vector.tobytes(), 
                combined_vector.tobytes(),
                metadata,
                image_id
            ))
            logger.info(f"Image vector updated successfully: ID {image_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to update image vector: {e}")
            raise
    
    def delete_image_vector(self, image_id):
        """Görsel vektörünü veritabanından sil"""
        query = "DELETE FROM image_vectors WHERE image_id = ?"
        
        try:
            result = self.execute_query(query, (image_id,))
            logger.info(f"Image vector deleted successfully: ID {image_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete image vector: {e}")
            raise
    
    def search_similar_images_db(self, query_vector, top_k=50):
        """Veritabanından benzer görselleri ara (cosine similarity)"""
        # Bu fonksiyon daha sonra implement edilecek
        # Şimdilik basit bir arama yapıyoruz
        query = """
        SELECT TOP (?) 
            image_id,
            image_path,
            image_name,
            upload_date,
            metadata
        FROM image_vectors
        ORDER BY upload_date DESC
        """
        
        try:
            results = self.execute_query(query, (top_k,))
            return results
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return []

# Global database manager instance
db_manager = DatabaseManager()

def get_db_manager():
    """Global database manager instance'ını döndür"""
    return db_manager
