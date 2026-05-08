import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="fa_kt")
    parser.add_argument("--emb_type", type=str, default="qidband")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--final_fc_dim", type=int, default=256)
    parser.add_argument("--final_fc_dim2", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--loss1", type=float, default=0.1)
    # parser.add_argument("--spec_loss_weight", type=float, default= 0.0005)
    # parser.add_argument("--loss2", type=float, default=0.5)
    # parser.add_argument("--loss3", type=float, default=0.5)
    
    parser.add_argument("--start", type=int, default=50)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--kernel_size1", type=int, default=5)
    parser.add_argument("--kernel_size2", type=int, default=5)
    parser.add_argument("--confidence_thresholds", type=str, default="[0.8, 0.6, 0.4]")

    parser.add_argument('--mamba_d_state', type=int, default=16)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=2)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)
