"""Build the small shared CloudFormation stack without a template dependency."""
from pathlib import Path


def ref(name):
    return {"Ref": name}


def attr(name, field):
    return {"Fn::GetAtt": [name, field]}


def sub(value):
    return {"Fn::Sub": value}


def trust(service):
    return {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": service},
                                                    "Action": "sts:AssumeRole"}]}


def statement(actions, resources="*", **extra):
    return {"Effect": "Allow", "Action": actions, "Resource": resources, **extra}


def policy(name, statements):
    return {"PolicyName": name, "PolicyDocument": {"Version": "2012-10-17", "Statement": statements}}


def template():
    tags = [{"Key": "BenchStack", "Value": ref("AWS::StackName")}]
    resources = {}
    def add(name, kind, properties, **extra):
        resources[name] = {"Type": "AWS::" + kind, "Properties": properties, **extra}
    add("Bucket", "S3::Bucket", {
        "BucketName": sub("${AWS::StackName}-${AWS::AccountId}-${AWS::Region}"),
        "BucketEncryption": {"ServerSideEncryptionConfiguration": [{"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]},
        "PublicAccessBlockConfiguration": {k: True for k in ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")},
        "VersioningConfiguration": {"Status": "Enabled"}, "Tags": tags,
        "LifecycleConfiguration": {"Rules": [{"Id": "abandoned-uploads", "Status": "Enabled",
                                                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}}]},
    }, DeletionPolicy="Retain", UpdateReplacePolicy="Retain")
    add("ScratchBucket", "S3::Bucket", {
        "BucketEncryption": {"ServerSideEncryptionConfiguration": [{"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]},
        "PublicAccessBlockConfiguration": {k: True for k in ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")},
        "Tags": tags,
        "LifecycleConfiguration": {"Rules": [{"Id": "disposable-ducklake", "Status": "Enabled",
            "Prefix": "scratch/", "ExpirationInDays": 2,
            "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}}]},
    }, DeletionPolicy="Retain", UpdateReplacePolicy="Retain")
    add("ScratchBucketPolicy", "S3::BucketPolicy", {"Bucket": ref("ScratchBucket"), "PolicyDocument": {
        "Version": "2012-10-17", "Statement": [{"Effect": "Deny", "Principal": "*", "Action": "s3:*",
        "Resource": [attr("ScratchBucket", "Arn"), sub("${ScratchBucket.Arn}/*")],
        "Condition": {"Bool": {"aws:SecureTransport": "false"}}}]}})
    add("BucketPolicy", "S3::BucketPolicy", {"Bucket": ref("Bucket"), "PolicyDocument": {
        "Version": "2012-10-17", "Statement": [{"Effect": "Deny", "Principal": "*", "Action": "s3:*",
        "Resource": [attr("Bucket", "Arn"), sub("${Bucket.Arn}/*")], "Condition": {"Bool": {"aws:SecureTransport": "false"}}}]}})
    add("Repository", "ECR::Repository", {"RepositoryName": ref("AWS::StackName"), "ImageTagMutability": "IMMUTABLE",
                                            "ImageScanningConfiguration": {"ScanOnPush": True}, "Tags": tags},
        DeletionPolicy="Retain", UpdateReplacePolicy="Retain")
    add("Vpc", "EC2::VPC", {"CidrBlock": "10.73.0.0/16", "EnableDnsSupport": True, "EnableDnsHostnames": True, "Tags": tags})
    add("Gateway", "EC2::InternetGateway", {"Tags": tags})
    add("GatewayAttachment", "EC2::VPCGatewayAttachment", {"VpcId": ref("Vpc"), "InternetGatewayId": ref("Gateway")})
    add("Routes", "EC2::RouteTable", {"VpcId": ref("Vpc"), "Tags": tags})
    add("InternetRoute", "EC2::Route", {"RouteTableId": ref("Routes"), "DestinationCidrBlock": "0.0.0.0/0", "GatewayId": ref("Gateway")}, DependsOn="GatewayAttachment")
    add("Subnet", "EC2::Subnet", {"VpcId": ref("Vpc"), "CidrBlock": "10.73.1.0/24",
                                   "AvailabilityZone": ref("AvailabilityZone"), "MapPublicIpOnLaunch": True, "Tags": tags})
    add("SubnetRoutes", "EC2::SubnetRouteTableAssociation", {"SubnetId": ref("Subnet"), "RouteTableId": ref("Routes")})
    add("SecurityGroup", "EC2::SecurityGroup", {"VpcId": ref("Vpc"), "GroupDescription": "Benchmark outbound access; no inbound ports",
                                                "SecurityGroupEgress": [{"IpProtocol": "-1", "CidrIp": "0.0.0.0/0"}], "Tags": tags})
    add("InstanceRole", "IAM::Role", {"RoleName": sub("${AWS::StackName}-instance"), "AssumeRolePolicyDocument": trust("ec2.amazonaws.com"),
        "ManagedPolicyArns": [sub("arn:${AWS::Partition}:iam::aws:policy/AmazonSSMManagedInstanceCore")],
        "Policies": [policy("artifacts", [
            statement(["s3:ListBucket"], attr("Bucket", "Arn")),
            statement(["s3:GetObject", "s3:PutObject"], sub("${Bucket.Arn}/runs/*")),
            statement(["s3:GetObject"], sub("${Bucket.Arn}/bundles/*")),
            statement(["s3:GetBucketVersioning"], attr("ScratchBucket", "Arn")),
            statement(["s3:ListBucket"], attr("ScratchBucket", "Arn"),
                      Condition={"StringLike": {"s3:prefix": "scratch/*"}}),
            statement(["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], sub("${ScratchBucket.Arn}/scratch/*")),
            statement(["ecr:GetAuthorizationToken"]),
            statement(["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"], attr("Repository", "Arn")),
        ])]})
    add("InstanceProfile", "IAM::InstanceProfile", {"Roles": [ref("InstanceRole")]})
    add("WatchdogRole", "IAM::Role", {"AssumeRolePolicyDocument": trust("lambda.amazonaws.com"), "Policies": [policy("stop-expired", [
        statement(["ec2:DescribeInstances"]),
        statement(["ec2:StopInstances"], sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:instance/*"),
                  Condition={"StringEquals": {"ec2:ResourceTag/BenchStack": ref("AWS::StackName")}}),
        statement(["logs:CreateLogStream", "logs:PutLogEvents"], sub("arn:${AWS::Partition}:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/lambda/${AWS::StackName}-watchdog:*")),
    ])]})
    add("WatchdogLog", "Logs::LogGroup", {"LogGroupName": sub("/aws/lambda/${AWS::StackName}-watchdog"), "RetentionInDays": 14})
    add("Watchdog", "Lambda::Function", {"FunctionName": sub("${AWS::StackName}-watchdog"), "Runtime": "python3.12",
        "Architectures": ["arm64"], "Handler": "index.handler", "Role": attr("WatchdogRole", "Arn"), "Timeout": 45,
        "Environment": {"Variables": {"STACK": ref("AWS::StackName")}},
        "Code": {"ZipFile": (Path(__file__).parent / "watchdog.py").read_text()}}, DependsOn="WatchdogLog")
    add("WatchdogRule", "Events::Rule", {"ScheduleExpression": "rate(1 minute)", "State": "ENABLED",
                                         "Targets": [{"Arn": attr("Watchdog", "Arn"), "Id": "watchdog"}]})
    add("WatchdogPermission", "Lambda::Permission", {"FunctionName": ref("Watchdog"), "Action": "lambda:InvokeFunction",
                                                     "Principal": "events.amazonaws.com", "SourceArn": attr("WatchdogRule", "Arn")})
    # Routine publishing/launching/retrieval can use this role instead of root.
    # Shared-stack administration remains a separate, infrequent operation.
    add("OperatorRole", "IAM::Role", {"RoleName": sub("${AWS::StackName}-operator"),
        "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
            "Principal": {"AWS": sub("arn:${AWS::Partition}:iam::${AWS::AccountId}:root")}, "Action": "sts:AssumeRole"}]},
        "Policies": [policy("operate-benchmark", [
            statement(["ec2:Describe*", "servicequotas:GetServiceQuota", "pricing:GetProducts", "ecr:GetAuthorizationToken"]),
            statement(["cloudformation:DescribeStacks"], sub("arn:${AWS::Partition}:cloudformation:${AWS::Region}:${AWS::AccountId}:stack/${AWS::StackName}/*")),
            statement(["ssm:GetParameter"], sub("arn:${AWS::Partition}:ssm:${AWS::Region}::parameter/aws/service/canonical/ubuntu/*")),
            statement(["s3:ListBucket"], attr("Bucket", "Arn")),
            statement(["s3:GetObject", "s3:PutObject"], sub("${Bucket.Arn}/*")),
            statement(["ecr:ListImages", "ecr:DescribeImages", "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload",
                       "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"], attr("Repository", "Arn")),
            statement(["iam:PassRole"], attr("InstanceRole", "Arn"), Condition={"StringEquals": {"iam:PassedToService": "ec2.amazonaws.com"}}),
            statement(["lambda:InvokeFunction"], attr("Watchdog", "Arn")),
            statement(["events:DescribeRule"], attr("WatchdogRule", "Arn")),
            statement(["ec2:RunInstances"], [
                sub("arn:${AWS::Partition}:ec2:${AWS::Region}::image/*"),
                sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:subnet/${Subnet}"),
                sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:security-group/${SecurityGroup}"),
                sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:network-interface/*")]),
            statement(["ec2:RunInstances"], [sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:instance/*"),
                                           sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:volume/*")],
                      Condition={"StringEquals": {"aws:RequestTag/BenchStack": ref("AWS::StackName")}}),
            statement(["ec2:CreateTags"], sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:*/*"),
                      Condition={"StringEquals": {"ec2:CreateAction": "RunInstances"}}),
            statement(["ec2:StopInstances", "ec2:TerminateInstances", "ec2:DeleteVolume"],
                      sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:*/*"),
                      Condition={"StringEquals": {"ec2:ResourceTag/BenchStack": ref("AWS::StackName")}}),
        ])]})
    outputs = {name: {"Value": value} for name, value in {
        "Bucket": ref("Bucket"), "ScratchBucket": ref("ScratchBucket"),
        "Repository": attr("Repository", "RepositoryUri"), "Subnet": ref("Subnet"),
        "SecurityGroup": ref("SecurityGroup"), "InstanceProfile": attr("InstanceProfile", "Arn"),
        "Watchdog": ref("Watchdog"), "WatchdogRule": ref("WatchdogRule"), "AvailabilityZone": ref("AvailabilityZone"),
        "OperatorRole": attr("OperatorRole", "Arn"),
    }.items()}
    return {"AWSTemplateFormatVersion": "2010-09-09", "Description": "Join pruning benchmark shared infrastructure",
            "Parameters": {"AvailabilityZone": {"Type": "AWS::EC2::AvailabilityZone::Name"}},
            "Resources": resources, "Outputs": outputs}
