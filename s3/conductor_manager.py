#https://dataplatform.aiml.apple.com/docs/conductor/apis
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
import random
import time

class ConductorManager:
    def __init__(self):
        conductor_endpoint = "https://conductor.data.apple.com"
        token = os.getenv('aws_access_key_id')
        user_name = os.getenv('aws_secret_access_key')
        self.conductor = boto3.client('s3', 
                                  endpoint_url=conductor_endpoint, 
                                  aws_access_key_id=token,
                                  aws_secret_access_key=user_name
                    )
        
    def create_bucket(self, bucket_name):
        return self.conductor.create_bucket(Bucket=bucket_name)
    
    def delete_bucket(self, bucket_name):
        objects = self.conductor.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                self.conductor.delete_object(Bucket=bucket_name, Key=obj['Key'])
            print(f"All objects in bucket '{bucket_name}' deleted.")

        self.conductor.delete_bucket(Bucket=bucket_name) 
    
    def list_objects(self, bucket_name, prefix=''):
        """List objects in the bucket."""
        try:
            response = self.conductor.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if 'Contents' in response:
                if prefix:
                    return [obj['Key'][len(prefix):].lstrip('/') for obj in response['Contents']]
                else:
                    return [obj['Key'] for obj in response['Contents']]
            else:
                return []
        except ClientError as e:
            print(f"Error listing objects: {e}")
            return []
        
    def delete_object(self, bucket_name, object_name):
        """Delete an object from the S3 bucket."""
        try:
            self.s3.delete_object(Bucket=bucket_name, Key=object_name)
            print(f"Object {object_name} deleted from bucket {bucket_name}.")
        except ClientError as e:
            print(f"Error deleting object: {e}")
        
    def upload_file(self, bucket_name, file_name, object_name=None):
        """Upload a file to the S3 bucket."""
        if object_name is None:
            object_name = file_name
        try:
            self.conductor.upload_file(file_name, bucket_name, object_name)
            print(f"File {file_name} uploaded to {bucket_name} as {object_name}.")
        except (NoCredentialsError, PartialCredentialsError) as e:
            print(f"Error uploading file: {e}")

    def download_file(self, bucket_name, object_name, file_name=None, max_retries=5, initial_delay=1, max_delay=60):
        """
        Download a file from the S3 bucket with retry mechanism and atomic file operations.
        """
        if file_name is None:
            file_name = object_name

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        # Create a temporary file in the same directory
        temp_file = f"{file_name}.tmp{random.randint(0, 999999):06d}"
        
        retries = 0
        while retries < max_retries:
            try:
                # Download to temporary file
                self.conductor.download_file(bucket_name, object_name, temp_file)
                
                # Atomic rename
                os.replace(temp_file, file_name)
                
                print(f"File {object_name} downloaded successfully to {file_name}")
                return
                
            except (NoCredentialsError, PartialCredentialsError) as e:
                print(f"Credential error downloading file: {e}")
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return
                
            except ClientError as e:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
                if e.response['Error']['Code'] == 'SlowDown':
                    delay = min(initial_delay * (2 ** retries) + random.uniform(0, 1), max_delay)
                    print(f"Throttling error. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    retries += 1
                else:
                    print(f"Error downloading file: {e}")
                    return
                    
            except Exception as e:
                print(f"Unexpected error downloading file: {e}")
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return
                
        # Clean up if we exceeded retries
        if os.path.exists(temp_file):
            os.unlink(temp_file)
