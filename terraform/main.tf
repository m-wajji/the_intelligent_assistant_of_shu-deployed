provider "aws" {
  region = var.aws_region
}

locals {
  key_path = "${path.module}/shu-agent-key.pub"
}

resource "aws_key_pair" "deploy" {
  key_name   = var.key_name
  public_key = file(local.key_path)
}

resource "aws_security_group" "swarm_sg" {
  name        = "swarm-sg"
  description = "Allow SSH, HTTP, API"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

}

resource "aws_instance" "swarm_mgr" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.deploy.key_name
  vpc_security_group_ids = [aws_security_group.swarm_sg.id]
   # Add root block device configuration
  root_block_device {
    volume_type           = "gp3"        # General Purpose SSD (recommended)
    volume_size           = 20           # Size in GB (increased from default 8GB)
    delete_on_termination = true         # Delete volume when instance terminates
    encrypted             = false        # Set to true if you want encryption
    
    tags = {
      Name = "swarm-manager-root-volume"
    }
  }
  tags = { Name = "swarm-manager" }
}

output "manager_ip" {
  value = aws_instance.swarm_mgr.public_ip
}