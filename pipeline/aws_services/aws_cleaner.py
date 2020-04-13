# To clean imagesets from false images

import boto3

__author__ = "cstur"

region_name = 'eu-central-1	'

class AWS_Cleaner:
    def __init__(self, testing=True):
        self.testing = testing
        if self.testing:
            self.endpoint_url = 'https://mturk-requester-sandbox.eu-central-1.amazonaws.com'
        else:
            self.endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

    aws_access_key_id = 'YOUR_ACCESS_ID'
    aws_secret_access_key = 'YOUR_SECRET_KEY'


    # Uncomment this line to use in production

    client = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # This will return $10,000.00 in the MTurk Developer Sandbox
    print(client.get_account_balance()['AvailableBalance'])