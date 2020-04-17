# To clean imagesets from false images
import datetime
import math
import os
import re
import uuid

import boto3
from botocore.exceptions import ClientError

import settings
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

    def create_task_with_batch(self, title: str, instructions: str, parameter_file, process_time_in_s: int = 30,
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

        parameters = []
        with open(parameter_file, "r") as fin:
            param_name= re.sub('[{}]', '', fin.readline().strip())
            for line in fin.read().splitlines()[1:]:
                parameters.append([{"Name": param_name, "Value": line}])

        for i in range(0, len(parameters)):
            # lifetime = how many seconds its available for workers until it is removed
            response = self.client.create_hit(MaxAssignments=workers_per_hit, LifetimeInSeconds=60 * 60 * 24 * 7,
                AssignmentDurationInSeconds=process_time_in_s, Reward=reward, Title=title,
                HITLayoutId="3Y5KTGTWK7O4WEXU4H5EDNH26UP5R9", HITLayoutParameters=parameters[i],
                Description=instructions, QualificationRequirements=worker_requirements)
            print(f"Created HIT ({i+1}/{len(parameters)})")

            # # The response included several fields that will be helpful later
            # hit_type_id = response['HIT']['HITTypeId']
            # hit_id = response['HIT']['HITId']
            # hit_total=response['HIT']["NumberOfAssignmentsPending"]+response['HIT']["NumberOfAssignmentsAvailable"]+response['HIT']["NumberOfAssignmentsAvailable"]
            # int_reward = int(reward.replace(".", ""))
            # print(f"Created HIT: {hit_id} ; Total cost: {workers_per_hit*hit_total*int_reward} cents ; {i}/{hits_to_create}")
            # print(f"You can work the HIT here: {self.preview}?groupId={hit_type_id}")
            # print(f"And see results here: {self.manage_url}")

        self.print_current_HITs()

    def upload_folder_to_s3(self, bucket_name: str, local_dir: str):
        s3 = boto3.client(service_name='s3',
            aws_access_key_id=credentials.AWSAccessKeyId, aws_secret_access_key=credentials.AWSSecretKey)

        bucket_name=str(re.sub(r'\W', '', bucket_name.lower())).replace("_", "")
        try:
            s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'eu-west-1'})
        except:
            pass

        # only upload files that are not yet in S3
        local_files=os.listdir(local_dir)
        cloud_files=[]
        response=s3.list_objects_v2(Bucket=bucket_name)
        if "Contents" in response.keys():
            for obj in response["Contents"]:
                cloud_files.append(obj["Key"])
        new_files = [file for file in local_files if file not in cloud_files]

        # uploads
        for count, file in enumerate(new_files):
            s3.upload_file(os.path.join(local_dir, file), bucket_name, file, ExtraArgs={'ACL': 'public-read'})
            print(f"Uploaded {file} to {bucket_name} ({count+1}/{len(new_files)}).")

        # creates csv
        response=s3.list_objects_v2(Bucket=bucket_name)
        with open(os.path.join(settings.root_dir, "pipeline", "aws_services", "s3_references", bucket_name+".csv"), "w+") as fout:
            if len(fout.readlines())==0:
                fout.write("image_name\n")
            for obj in response["Contents"]:
                fout.write(obj["Key"]+"\n")

    def print_current_HITs(self):
        num_hits=self.client.list_hits()["NumResults"]
        print(f"See results here: {self.manage_url}")
        print(f"Currently {num_hits} HITs are online.")
        for hit in self.client.list_hits()["HITs"]:
            hit_id=hit["HITId"]
            hit_title=hit["Title"]
            hit_type_id=hit['HITTypeId']
            hit_completed=hit["NumberOfAssignmentsAvailable"]
            hit_total=hit["NumberOfAssignmentsPending"]+hit["NumberOfAssignmentsAvailable"]+hit_completed
            int_reward=int(hit["Reward"].replace(".", ""))
            print(f"Title: {hit_title} ; HITId: {hit_id} ; {self.preview}?groupId={hit_type_id}; ; Total cost: {hit_total*int_reward} cents ; ({hit_completed}/{hit_total})")