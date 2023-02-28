import mujoco_py as mj
import numpy as np
import torch
from copy import deepcopy

nwarmup = 3
eps = 1e-6


def calculate_gradients(agent, data_snapshot, next_state, reward):
    # Defining m and d just for shorter notations
    m = agent.model
    d = agent.data

    # Dynamics gradients
    dsdctrl = np.empty((m.nq + m.nv, m.nu))
    dsdqpos = np.empty((m.nq + m.nv, m.nq))
    dsdqvel = np.empty((m.nq + m.nv, m.nv))

    # Reward gradients
    drdctrl = np.empty((1, m.nu))
    drdqpos = np.empty((1, m.nq))
    drdqvel = np.empty((1, m.nv))

    # Get number of steps (must be >=2 for muscles)
    nsteps = 1

    # Get state from the forward pass
    if nsteps > 1:
        agent.set_snapshot(data_snapshot)
        for _ in range(nsteps):
            info = agent.step(d.ctrl.copy())
        qpos_fwd = info[0][:agent.model.nq]
        qvel_fwd = info[0][agent.model.nq:]
        reward = info[1]
    else:
        qpos_fwd = next_state[:agent.model.nq]
        qvel_fwd = next_state[agent.model.nq:]

    # finite-difference over control values
    for i in range(m.nu):

        # Initialise simulation
        agent.set_snapshot(data_snapshot)

        # Perturb control
        d.ctrl[i] += eps

        # Step with perturbed simulation
        for _ in range(nsteps):
            info = agent.step(d.ctrl.copy())

        # Compute gradient of state wrt control
        dsdctrl[:m.nq, i] = (d.qpos - qpos_fwd) / eps
        dsdctrl[m.nq:, i] = (d.qvel - qvel_fwd) / eps

        # Compute gradient of reward wrt to control
        drdctrl[0, i] = (info[1] - reward) / eps

    # finite-difference over velocity
    for i in range(m.nv):

        # Initialise simulation
        agent.set_snapshot(data_snapshot)

        # Perturb velocity
        d.qvel[i] += eps

        # Step with perturbed simulation
        for _ in range(nsteps):
            info = agent.step(d.ctrl)

        # Compute gradient of state wrt qvel
        dsdqvel[:m.nq, i] = (d.qpos - qpos_fwd) / eps
        dsdqvel[m.nq:, i] = (d.qvel - qvel_fwd) / eps

        # Compute gradient of reward wrt qvel
        drdqvel[0, i] = (info[1] - reward) / eps

    # finite-difference over position
    for i in range(m.nq):

        # Initialise simulation
        agent.set_snapshot(data_snapshot)

        # Get joint id for this dof
        jid = m.dof_jntid[i]

        # Get quaternion address and dof position within quaternion (-1: not in quaternion)
        quatadr = -1
        dofpos = 0
        if m.jnt_type[jid] == mj.const.JNT_BALL:
            quatadr = m.jnt_qposadr[jid]
            dofpos = i - m.jnt_dofadr[jid]
        elif m.jnt_type[jid] == mj.const.JNT_FREE and i >= m.jnt_dofadr[jid] + 3:
            quatadr = m.jnt_qposadr[jid] + 3
            dofpos = i - m.jnt_dofadr[jid] - 3

        # Apply quaternion or simple perturbation
        if quatadr >= 0:
            angvel = np.array([0., 0., 0.])
            angvel[dofpos] = eps
            mj.functions.mju_quatIntegrate(d.qpos + quatadr, angvel, 1)
        else:
            d.qpos[m.jnt_qposadr[jid] + i - m.jnt_dofadr[jid]] += eps

        # Step simulation with perturbed position
        for _ in range(nsteps):
            info = agent.step(d.ctrl)

        # Compute gradient of state wrt qpos
        dsdqpos[:m.nq, i] = (d.qpos - qpos_fwd) / eps
        dsdqpos[m.nq:, i] = (d.qvel - qvel_fwd) / eps

        # Compute gradient of reward wrt qpos
        drdqpos[0, i] = (info[1] - reward) / eps

    # Set dynamics gradients
    agent.dynamics_gradients = {"state": np.concatenate((dsdqpos, dsdqvel), axis=1), "action": dsdctrl}

    # Set reward gradients
    agent.reward_gradients = {"state": np.concatenate((drdqpos, drdqvel), axis=1), "action": drdctrl}

    return


def mj_gradients_factory(agent, mode):
    @agent.gradient_wrapper(mode)
    def mj_gradients(data_snapshot, next_state, reward):
        calculate_gradients(agent, data_snapshot, next_state, reward)

    return mj_gradients
