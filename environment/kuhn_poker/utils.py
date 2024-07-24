import torch
from learning.model import LatentPolicy


def get_probs_and_return(args, policy: LatentPolicy, policies, history, device, get_idx=None):
    num_policies = len(policies)
    num_processes = num_policies
    if get_idx is None:
        all_policy_indices = torch.arange(6 * num_processes) % num_policies
    else:
        all_policy_indices = torch.zeros(6, dtype=torch.long)
        num_processes = 1
    with torch.no_grad():
        # all_probs = []
        # all_qvalues = []
        inputs = torch.zeros(6 * num_processes, 13).to(device)
        masks = torch.ones(6 * num_processes, 1).to(device)
        if history is not None:
            all_history = history.get_all_current()[0] if policy.latent_training_mode and history is not None else None
            if get_idx is not None:
                sizes, obs, act = all_history
                sizes = [sizes[get_idx]]
                obs = [obs[get_idx]]
                act = [act[get_idx]]
                all_history = (sizes, obs, act)
            all_latents, all_params = policy.encoder(all_history, None)
        else:
            all_latents = all_params = None

        for i in range(6):
            inputs[i * num_processes:(i + 1) * num_processes, i // 3] = 1.0
            inputs[i * num_processes:(i + 1) * num_processes, 7 + (i % 3)] = 1.0
        action_dist, _, _ = policy.get_action_dist(
            inputs, None, masks, all_policy_indices, None, all_latents, all_params
        )
        all_probs = action_dist.probs[:, 1].cpu().reshape(6, num_processes).transpose(0, 1)
        # if args.algo == 'dqn' or args.value_obj:
        #     _, qvalue = policy.get_value(inputs, all_policy_indices, None, True)
        #     qvalue = qvalue.cpu().reshape(num_processes // num_policies, num_policies, 2).mean(dim=0)
        #     all_qvalues.append(qvalue)
        # all_probs = torch.stack(all_probs, dim=1)
        # if args.algo == 'dqn' or args.value_obj:
        #     all_qvalues = torch.stack(all_qvalues, dim=1)
        # else:
        #     all_qvalues = None
        if get_idx is None:
            all_returns = [policies[i % num_policies].get_return_complex(all_probs[i].tolist()) for i in range(len(all_probs))]
            all_det_returns = [policies[i % num_policies].get_return_complex([1.0 if v >= 0.5 else 0.0 for v in all_probs[i].tolist()])
                               for i in range(len(all_probs))]
        else:
            all_returns = [policies[get_idx % num_policies].get_return_complex(all_probs[0].tolist())]
            all_det_returns = [
                policies[get_idx % num_policies].get_return_complex([1.0 if v >= 0.5 else 0.0 for v in all_probs[0].tolist()])
            ]
    return all_probs, None, all_returns, all_det_returns
