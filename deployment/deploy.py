"""Deploy registered model versions to an Azure ML managed online endpoint.

    python deployment/deploy.py create    --endpoint NAME --version 1             # Q7a: endpoint + 'blue' (100% traffic)
    python deployment/deploy.py test      --endpoint NAME [--deployment green]    # Q7c: send sample_request.json
    python deployment/deploy.py add-green --endpoint NAME --version 2 --traffic 30  # Q7e: 'green' gets 30%
    python deployment/deploy.py show      --endpoint NAME                          # scoring URI, traffic split
    python deployment/deploy.py secrets   --endpoint NAME   # write .streamlit/secrets.toml for the Streamlit app
    python deployment/deploy.py delete    --endpoint NAME   # endpoints bill until deleted

Endpoint names must be unique within the Azure region (3-32 chars, letters/digits/hyphens).
Deployments serve through score.py and conda.yaml in this folder. Pass --no_code to use
Azure's no-code MLflow deployment instead. Its scoring script imports azureml-ai-monitoring,
which train.py adds to every saved model's requirements.
"""
import argparse
import sys
from pathlib import Path

from azure.ai.ml.entities import (CodeConfiguration, Environment,
                                  ManagedOnlineDeployment, ManagedOnlineEndpoint)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE.parent / "src"))
from register import MODEL_NAME  # noqa: E402
from workspace import ROOT, get_ml_client  # noqa: E402

SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"


def build_deployment(name, endpoint, version, instance_type, custom_scoring):
    extra = {}
    if custom_scoring:
        extra = dict(
            environment=Environment(
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
                conda_file=str(HERE / "conda.yaml"),
            ),
            code_configuration=CodeConfiguration(code=str(HERE), scoring_script="score.py"),
        )
    return ManagedOnlineDeployment(
        name=name,
        endpoint_name=endpoint,
        model=f"azureml:{MODEL_NAME}:{version}",
        instance_type=instance_type,
        instance_count=1,
        **extra,
    )


def set_traffic(ml_client, endpoint_name, traffic):
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = traffic
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Traffic for {endpoint_name}: {traffic}")


def create(ml_client, args):
    endpoint = ManagedOnlineEndpoint(
        name=args.endpoint,
        description="Telecom customer churn prediction (random forest)",
        auth_mode="key",
    )
    print(f"Creating endpoint {args.endpoint} ...")
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"Creating deployment 'blue' with {MODEL_NAME} v{args.version} (takes several minutes) ...")
    ml_client.online_deployments.begin_create_or_update(
        build_deployment("blue", args.endpoint, args.version, args.instance_type, not args.no_code)
    ).result()
    set_traffic(ml_client, args.endpoint, {"blue": 100})
    test(ml_client, args)


def add_green(ml_client, args):
    print(f"Creating deployment 'green' with {MODEL_NAME} v{args.version} (takes several minutes) ...")
    ml_client.online_deployments.begin_create_or_update(
        build_deployment("green", args.endpoint, args.version, args.instance_type, not args.no_code)
    ).result()
    set_traffic(ml_client, args.endpoint, {"blue": 100 - args.traffic, "green": args.traffic})


def test(ml_client, args):
    target = f"deployment '{args.deployment}'" if args.deployment else "traffic-split routing"
    print(f"Sending {HERE / 'sample_request.json'} via {target}")
    response = ml_client.online_endpoints.invoke(
        endpoint_name=args.endpoint,
        request_file=str(HERE / "sample_request.json"),
        deployment_name=args.deployment,
    )
    print("Response:", response)


def show(ml_client, args):
    endpoint = ml_client.online_endpoints.get(args.endpoint)
    print("Scoring URI:", endpoint.scoring_uri)
    print("Traffic:", endpoint.traffic)
    for deployment in ml_client.online_deployments.list(args.endpoint):
        print(f"Deployment {deployment.name}: model={deployment.model}, state={deployment.provisioning_state}")


def secrets(ml_client, args):
    endpoint = ml_client.online_endpoints.get(args.endpoint)
    key = ml_client.online_endpoints.get_keys(args.endpoint).primary_key
    SECRETS_FILE.parent.mkdir(exist_ok=True)
    SECRETS_FILE.write_text(f'scoring_uri = "{endpoint.scoring_uri}"\nendpoint_key = "{key}"\n')
    print(f"Wrote scoring URI and key to {SECRETS_FILE.relative_to(ROOT)} (git-ignored)")


def delete(ml_client, args):
    print(f"Deleting endpoint {args.endpoint} and all its deployments ...")
    ml_client.online_endpoints.begin_delete(name=args.endpoint).result()
    print("Deleted")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["create", "add-green", "test", "show", "secrets", "delete"]:
        p = sub.add_parser(name)
        p.add_argument("--endpoint", required=True)
        if name in ("create", "add-green"):
            p.add_argument("--version", required=True, help=f"{MODEL_NAME} version to deploy")
            # Azure ML reserves ~20% extra per deployment, so one 2-vCPU instance holds
            # 4 vCPUs of quota. Free-trial quota is 6 vCPUs per VM family, so blue and
            # green use different families.
            p.add_argument("--instance_type",
                           default="Standard_DS2_v2" if name == "create" else "Standard_F2s_v2")
            p.add_argument("--no_code", action="store_true",
                           help="use Azure's no-code MLflow deployment instead of score.py")
        if name == "add-green":
            p.add_argument("--traffic", type=int, default=30, help="percent of traffic for green")
        if name in ("create", "test"):
            p.add_argument("--deployment", default=None, help="target one deployment, bypassing the split")
    args = parser.parse_args()

    commands = {"create": create, "add-green": add_green, "test": test,
                "show": show, "secrets": secrets, "delete": delete}
    commands[args.command](get_ml_client(), args)


if __name__ == "__main__":
    main()
