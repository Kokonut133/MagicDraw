# To clean imagesets from false images
import datetime

import boto3
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
                client = session.client('mturk',
                    region_name="us-east-1",
                    aws_access_key_id=credentials.AWSAccessKeyId,
                    aws_secret_access_key=credentials.AWSSecretKey,
                    endpoint_url="https://mturk-requester-sandbox.us-east-1.amazonaws.com/",
                )
                manage_url = "https://requestersandbox.mturk.com/mturk/manageHITs"

            else:
                client = session.client('mturk',
                    region_name="us-east-1",
                    aws_access_key_id=credentials.AWSAccessKeyId,
                    aws_secret_access_key=credentials.AWSSecretKey,
                    endpoint_url="https://mturk-requester.us-east-1.amazonaws.com",
                )
                manage_url = "https://requester.mturk.com/mturk/manageHITs"
            print("Account balance is \n", client.get_account_balance())

        except ClientError as e:
            print(e)

    def create_task(self, instructions:str, min_approval_rating:int=90):
        # @param instructions

        worker_requirements = [{
            'QualificationTypeId': '000000000000000000L0',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [min_approval_rating],
            'RequiredToPreview': True,
        }]
        #
        # # Create the HIT
        # response = client.create_hit(
        #     MaxAssignments=3,
        #     LifetimeInSeconds=600,
        #     AssignmentDurationInSeconds=600,
        #     Reward=mturk_environment['reward'],
        #     Title='Answer a simple question',
        #     Keywords='question, answer, research',
        #     Description='Answer a simple question. Created from mturk-code-samples.',
        #     Question=question_sample,
        #     QualificationRequirements=worker_requirements,
        # )
        #
        # # The response included several fields that will be helpful later
        # hit_type_id = response['HIT']['HITTypeId']
        # hit_id = response['HIT']['HITId']
        # print
        # "\nCreated HIT: {}".format(hit_id)
        #
        # print
        # "\nYou can work the HIT here:"
        # print
        # mturk_environment['preview'] + "?groupId={}".format(hit_type_id)
        #
        # print
        # "\nAnd see results here:"
        # print
        # mturk_environment['manage']