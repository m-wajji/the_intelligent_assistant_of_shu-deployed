terraform {
  backend "s3" {
    bucket       = "wajahat-terraform-state-bucket"
    key          = "shu-agent-swarm/terraform.tfstate"
    region       = "us-east-1"    
    encrypt      = true
    use_lockfile = true             
  }
}