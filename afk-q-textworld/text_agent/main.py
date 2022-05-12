import datetime
import os
import sys
import copy
import time
import json
import numpy as np
from pytz import timezone
from os.path import join as pjoin

sys.path.insert(0, os.path.abspath('../textworld'))  # HACK
import generic
from agent import CustomAgent
from tw_cooking_afk_env import register_tw_cooking_afk
import textworld
import gym


def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()

    infos = textworld.EnvInfos(description=True, inventory=True, admissible_commands=True, facts=True, score=True, objective=True,
                               extras=["ans", "seed", "walkthrough"])
    env_id = register_tw_cooking_afk(
        settings=config["tw"],
        request_infos=infos,
        batch_size=config["training"]["batch_size"],
        max_episode_steps=config["training"]["max_nb_steps_per_episode"],
        asynchronous=config["training"]["parallel"]
    )

    env = gym.make(env_id)
    agent = CustomAgent(config)
    seeds = env.seed(config["tw"]["seed"])

    # visdom
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        score_win = None
        q_value_win = None
        step_win = None
        viz_train_score, viz_train_reward, viz_eval_score = [], [], []
        viz_q_value = []
        viz_train_step, viz_eval_step = [], []

    step_in_total = 0
    batch_no = 0
    episode_no = 0
    running_avg_score = generic.HistoryScoreCache(capacity=50)
    running_avg_reward = generic.HistoryScoreCache(capacity=50)
    running_avg_step = generic.HistoryScoreCache(capacity=50)
    running_avg_q_value = generic.HistoryScoreCache(capacity=50)
    running_avg_loss = generic.HistoryScoreCache(capacity=50)

    output_dir = "."
    data_dir = "."
    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_train_reward_so_far = 0.0
    # load model from checkpoint

    if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
        print("checkpoint already exist.")
        # exit(0)

    if agent.load_pretrained:
        if os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(
                data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()

    while(True):
        if episode_no > agent.max_episode:
            break

        obs, infos = env.reset()
        # print("-= Games #{} =-".format(infos["extra.seed"]))
        batch_size = len(obs)
        report = agent.report_frequency > 0 and (episode_no % agent.report_frequency <= max(
            episode_no - batch_size, 0) % agent.report_frequency)
        if report:
            print("====================================================================================", episode_no)
            print("-- task: %s" % (infos["objective"][0]))

        agent.train()
        agent.init(obs, infos)

        previous_dynamics = None
        tmp_replay_buffer = []
        print_cmds = []
        dones_prev_step = None

        act_randomly = False if agent.noisy_net else episode_no < agent.learn_start_from_this_episode

        quest_token_ids = agent.get_input_token_ids(infos["objective"])
        for step_id in range(agent.max_nb_steps_per_episode):
            if step_id == 0:
                obs = [" ".join([_obs, _inv])
                       for _obs, _inv in zip(obs, infos["inventory"])]
            else:
                obs = [" ".join([_obs, _des, _inv]) for _obs, _des, _inv in zip(
                    obs, infos["description"], infos["inventory"])]

            # generate commands
            if agent.noisy_net:
                agent.reset_noise()  # Draw a new set of noisy weights
            commands, replay_info, current_dynamics = agent.act(
                obs, infos, quest_token_ids, previous_dynamics, random=act_randomly)

            tmp_replay_buffer.append(replay_info)
            obs, rewards, dones, infos = env.step(commands)
            previous_dynamics = current_dynamics

            if agent.noisy_net and step_in_total % agent.update_per_k_game_steps == 0:
                agent.reset_noise()  # Draw a new set of noisy weights

            if episode_no >= agent.learn_start_from_this_episode and step_in_total % agent.update_per_k_game_steps == 0:
                interaction_loss, interaction_q_value = agent.update_interaction()
                if interaction_loss is not None:
                    running_avg_loss.push(interaction_loss)
                    running_avg_q_value.push(interaction_q_value)

            step_in_total += 1
            if step_in_total % 2000000 == 0:
                agent.save_model_to_path(
                    output_dir + "/" + agent.experiment_tag + "_step_" + str(step_in_total) + "_model.pt")

            if dones_prev_step is not None:
                for b in range(batch_size):
                    agent.still_running[b] *= (1.0 - float(dones_prev_step[b]))
            dones_prev_step = copy.deepcopy(dones)

            posneg_rewards = list(rewards)
            if step_id < agent.max_nb_steps_per_episode - 1:
                for b in range(batch_size):
                    if float(dones[b]) == 1 and posneg_rewards[b] == 0:
                        posneg_rewards[b] = -1.0

            tmp_replay_buffer[-1].append(copy.deepcopy(rewards))
            tmp_replay_buffer[-1].append(copy.deepcopy(posneg_rewards))
            tmp_replay_buffer[-1].append(copy.deepcopy(agent.still_running))

            print_cmds.append(commands[0] if agent.still_running[0] else "--")
            if np.sum(agent.still_running) == 0:
                break

        if report:
            print(" / ".join(print_cmds).encode('utf-8'))

        kg_expanded = np.array([item[-4] for item in tmp_replay_buffer])  # step x batch
        # at every step, this value means "did the previous query success?"
        kg_expanded = np.concatenate(
            [kg_expanded[1:], kg_expanded[:1]], 0)  # step x batch
        game_rewards = np.array([item[-3] for item in tmp_replay_buffer])  # step x batch
        game_posneg_rewards = np.array([item[-2] for item in tmp_replay_buffer])  # step x batch
        masks = np.array([item[-1] for item in tmp_replay_buffer])  # step x batch

        # game rewards are accumulated (there might be 1's in paddings)
        # 0 0 0 0 1 1 1 -->
        # 0 0 0 0 1 0 0
        for i in range(len(game_rewards) - 1, -1, -1):
            if i > 0:
                game_rewards[i] = game_rewards[i] - game_rewards[i - 1]
                game_posneg_rewards[i] = game_posneg_rewards[i] - game_posneg_rewards[i - 1]

        # small positive reward whenever the agent expands kg
        if agent.setting in ["ours", "ours_note"]:
            if agent.use_negative_reward:
                merged_rewards = (agent.kg_expansion_reward *
                                kg_expanded + game_posneg_rewards) * masks
            else:
                merged_rewards = (agent.kg_expansion_reward *
                                kg_expanded + game_rewards) * masks
        else:
            if agent.use_negative_reward:
                merged_rewards = game_posneg_rewards * masks
            else:
                merged_rewards = game_rewards * masks
        merged_rewards_pt = generic.to_pt(merged_rewards, False, "float")
        overall_scores = np.sum(game_rewards, 0)  # batch
        overall_rewards = np.sum(merged_rewards, 0)  # batch
        overall_steps = np.sum(masks, 0)  # batch
        if report:
            print("rewards: ", str(merged_rewards[:, 0]))

        # push experience into replay buffer
        for b in range(batch_size):
            is_prior = overall_scores[b] > 0.0
            mem = []
            for i in range(len(tmp_replay_buffer)):
                if masks[i][b] == 0.0:
                    break
                description_in_cache, quest_in_cache, verb_indices_in_cache, \
                    adj_indices_in_cache, noun_indices_in_cache, nodes_in_cache, \
                    vocab_mask_ids_in_cache, _, _, _, _ = tmp_replay_buffer[i]
                mem.append([copy.copy(description_in_cache[b]),
                            copy.copy(quest_in_cache[b]),
                            copy.copy(verb_indices_in_cache[b]),
                            copy.copy(adj_indices_in_cache[b]),
                            copy.copy(noun_indices_in_cache[b]),
                            copy.copy(nodes_in_cache[b]) if nodes_in_cache is not None else None,
                            copy.copy([vocab_mask_ids_in_cache[0][b], vocab_mask_ids_in_cache[1][b], vocab_mask_ids_in_cache[2][b]]),
                            copy.copy(merged_rewards_pt[i][b])])
            agent.replay_memory.push(is_prior, mem)

        running_avg_score.push(np.mean(overall_scores))
        running_avg_reward.push(np.mean(overall_rewards))
        running_avg_step.push(np.mean(overall_steps))

        # finish game
        agent.finish_of_episode(episode_no, batch_no, batch_size)

        time_2 = datetime.datetime.now()
        eastern_time = datetime.datetime.now(
            timezone('US/Eastern')).strftime("%b %d %Y %H:%M:%S")
        if report:
            print("Episode/Step: {:s}/{:s} | {:s} | seed:{:s} | time spent: {:s} | loss: {:2.3f} | qvalue: {:2.3f} | score/reward: {:2.3f}/{:2.3f} | steps: {:2.3f}".format(str(episode_no), str(step_in_total), eastern_time, str(infos["extra.seed"][0]), str(
                time_2 - time_1).rsplit(".")[0], running_avg_loss.get_avg(),  running_avg_q_value.get_avg(), running_avg_score.get_avg(), running_avg_reward.get_avg(), running_avg_step.get_avg()))

        if not report or episode_no < agent.learn_start_from_this_episode:
            episode_no += batch_size
            batch_no += 1
            continue

        # save model
        if running_avg_reward.get_avg() >= best_train_reward_so_far:
            best_train_reward_so_far = running_avg_reward.get_avg()
            agent.save_model_to_path(
                output_dir + "/" + agent.experiment_tag + "_model.pt")
        # evaluate
        eval_scores, eval_steps = 0.0, 0.0
        if agent.run_eval:
            pass
            # eval_qa_acc, eval_ig_acc, eval_used_steps = evaluate.evaluate(env, agent, "valid")
            # env.split_reset("train")

        # plot using visdom
        if config["general"]["visdom"]:
            viz_train_score.append(running_avg_score.get_avg())
            viz_train_reward.append(running_avg_reward.get_avg())
            viz_q_value.append(running_avg_q_value.get_avg())
            viz_train_step.append(running_avg_step.get_avg())
            viz_eval_score.append(eval_scores)
            viz_eval_step.append(eval_steps)
            viz_x = np.arange(len(viz_train_score)).tolist()

            if score_win is None:
                score_win = viz.line(X=viz_x, Y=viz_train_score,
                                     opts=dict(title=agent.experiment_tag + "_scores"), name="score")
                viz.line(X=viz_x, Y=viz_train_reward,
                         win=score_win, update='append', name="reward")
                viz.line(X=viz_x, Y=viz_eval_score,
                         win=score_win, update='append', name="eval score")
            else:
                viz.line(X=[len(viz_train_score) - 1], Y=[viz_train_score[-1]],
                         win=score_win, update='append', name="score")
                viz.line(X=[len(viz_train_reward) - 1], Y=[viz_train_reward[-1]],
                         win=score_win, update='append', name="reward")
                viz.line(X=[len(viz_eval_score) - 1], Y=[viz_eval_score[-1]],
                         win=score_win, update='append', name="eval score")

            if q_value_win is None:
                q_value_win = viz.line(X=viz_x, Y=viz_q_value,
                                       opts=dict(
                                           title=agent.experiment_tag + "_q_value"),
                                       name="q value")
            else:
                viz.line(X=[len(viz_q_value) - 1], Y=[viz_q_value[-1]],
                         win=q_value_win, update='append', name="q value")

            if step_win is None:
                step_win = viz.line(X=viz_x, Y=viz_train_step,
                                    opts=dict(
                                        title=agent.experiment_tag + "_step"),
                                    name="train step")
                viz.line(X=viz_x, Y=viz_eval_step,
                         win=step_win, update='append', name="eval step")
            else:
                viz.line(X=[len(viz_train_step) - 1], Y=[viz_train_step[-1]],
                         win=step_win,
                         update='append', name="train step")
                viz.line(X=[len(viz_eval_step) - 1], Y=[viz_eval_step[-1]],
                         win=step_win, update='append', name="eval step")

        # write accucacies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                         "train score": str(running_avg_score.get_avg()),
                         "train reward": str(running_avg_reward.get_avg()),
                         "train step": str(running_avg_step.get_avg()),
                         "q value": str(running_avg_q_value.get_avg()),
                         "eval score": str(eval_scores),
                         "eval step": str(eval_steps)})
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()
        episode_no += batch_size
        batch_no += 1


if __name__ == '__main__':
    train()
