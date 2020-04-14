# To clean imagesets from false images
import datetime
import os
import re

import boto3
from boto3.s3.inject import bucket_upload_file
from botocore.exceptions import ClientError
import third_party.aws_keys as credentials

__author__ = "cstur"


class AWS_Imageprocessor:
    def __init__(self, testing=True):
        self.testing = testing
        session = boto3.Session()

        # us east 1 seems the only place for mturk
        try:
            if self.testing:
                self.client = session.client('mturk', region_name="us-east-1",
                    aws_access_key_id=credentials.AWSAccessKeyId, aws_secret_access_key=credentials.AWSSecretKey,
                    endpoint_url="https://mturk-requester-sandbox.us-east-1.amazonaws.com/", )
                self.manage_url = "https://requestersandbox.mturk.com/mturk/manageHITs"
                self.preview = "https://workersandbox.mturk.com/mturk/preview"

            else:
                self.client = session.client('mturk', region_name="us-east-1",
                    aws_access_key_id=credentials.AWSAccessKeyId, aws_secret_access_key=credentials.AWSSecretKey,
                    endpoint_url="https://mturk-requester.us-east-1.amazonaws.com", )
                self.manage_url = "https://requester.mturk.com/mturk/manageHITs"
                self.preview = "https://www.mturk.com/mturk/preview"

            print("Account balance is \n", self.client.get_account_balance())

        except ClientError as e:
            print(e)

    def create_task(self, title: str, instructions: str, question_file: str, process_time_in_s: int = 30,
                    reward: str = "0.01", min_approval_rating: int = 90, workers_per_hit: int = 1):
        """
        Args:
            title(str): Name for the Task.
            instructions(str): A string giving instructions to mturkers what to do.
            min_approval_rating(int): Minimum approval rating of a mturker to be allowed to this task. 0-100.
            reward(str): Reward per HIT in the form of a string. "0.01" -> 1 cent
            workers_per_hit(int): How many people should look at a single task.
            process_time_in_s(int): Time per task in seconds. 30seconds minimum by amazon.
        """

        worker_requirements = [{'QualificationTypeId': '000000000000000000L0', 'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [min_approval_rating], 'RequiredToPreview': True, }]

        question = open(question_file, "r").read()

        response = self.client.create_hit(MaxAssignments=workers_per_hit, LifetimeInSeconds=process_time_in_s,
            AssignmentDurationInSeconds=process_time_in_s, Reward=reward, Title=title, Question=question,
            Description=instructions, QualificationRequirements=worker_requirements, )

        # The response included several fields that will be helpful later
        hit_type_id = response['HIT']['HITTypeId']
        hit_id = response['HIT']['HITId']
        print("Created HIT: {}".format(hit_id))
        print("You can work the HIT here:")
        print(self.preview + "?groupId={}".format(hit_type_id))
        print("And see results here:")
        print(self.preview)

    def upload_folder_to_s3(self, bucket_name: str, local_dir: str):
        s3 = boto3.client(service_name='s3',
            aws_access_key_id=credentials.AWSAccessKeyId, aws_secret_access_key=credentials.AWSSecretKey)

        bucket_name=str(re.sub(r'\W', '', bucket_name.lower())).replace("_", "-")
        try:
            s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'eu-west-1'})
        except:
            pass

        # only upload files that are not yet in S3
        local_files=os.listdir(local_dir)
        cloud_files=[]
        response=s3.list_objects_v2(Bucket=bucket_name)
        for obj in response["Contents"]:
            cloud_files.append(obj["Key"])
        new_files = [file for file in local_files if file not in cloud_files]

        for count, file in enumerate(new_files):
            s3.upload_file(os.path.join(local_dir, file), bucket_name, file)
            print(f"Uploaded {file} to {bucket_name} ({count+1}/{len(new_files)}).")
