import time
import re 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import streamlit as st 
import streamlit.components.v1 as components

# Simulator options.
options = {}
options['FIG_SIZE'] = [7,7]
options['OBSTACLES'] = True

def sim_run(options, MPC, horizon, goal1, goal2, obs_pos):
    start = time.process_time()
    # Simulator Options
    FIG_SIZE = options['FIG_SIZE'] # [Width, Height]
    OBSTACLES = options['OBSTACLES']

    mpc = MPC(horizon, goal1, goal2, obs_pos)

    num_inputs = 2
    u = np.zeros(mpc.horizon*num_inputs)
    bounds = []

    # Set bounds for inputs bounded optimization.
    for i in range(mpc.horizon):
        bounds += [[-1, 1]]
        bounds += [[-0.8, 0.8]]

    ref_1 = mpc.reference1
    ref_2 = mpc.reference2
    ref = ref_1

    state_i = np.array([[0,0,0,0]])
    u_i = np.array([[0,0]])
    sim_total = 200
    predict_info = [state_i]

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for i in range(1,sim_total+1):
        u = np.delete(u,0)
        u = np.delete(u,0)
        u = np.append(u, u[-2])
        u = np.append(u, u[-2])
        start_time = time.time()

        # Non-linear optimization.
        u_solution = minimize(mpc.cost_function, u, (state_i[-1], ref),
                                method='SLSQP',
                                bounds=bounds,
                                tol = 1e-5)
        # print('Step ' + str(i) + ' of ' + str(sim_total) + '   Time ' + str(round(time.time() - start_time,5)))
        u = u_solution.x
        y = mpc.plant_model(state_i[-1], mpc.dt, u[0], u[1])
        if (i > 100 and ref_2 != None):
            ref = ref_2
        predicted_state = np.array([y])
        for j in range(1, mpc.horizon):
            predicted = mpc.plant_model(predicted_state[-1], mpc.dt, u[2*j], u[2*j+1])
            predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        predict_info += [predicted_state]
        state_i = np.append(state_i, np.array([y]), axis=0)
        u_i = np.append(u_i, np.array([(u[0], u[1])]), axis=0)
        
        my_bar.progress(i//2, text=progress_text)


    ###################
    # SIMULATOR DISPLAY

    # Total Figure
    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
    gs = gridspec.GridSpec(8,8)

    # Elevator plot settings.
    ax = fig.add_subplot(gs[:8, :8])

    plt.xlim(-3, 17)
    ax.set_ylim([-3, 17])
    plt.xticks(np.arange(0,11, step=2))
    plt.yticks(np.arange(0,11, step=2))
    plt.title('MPC 2D')

    # Time display.
    time_text = ax.text(6, 0.5, '', fontsize=15)

    # Main plot info.
    car_width = 1.0
    patch_car = mpatches.Rectangle((0, 0), car_width, 2.5, fc='k', fill=False)
    patch_goal = mpatches.Rectangle((0, 0), car_width, 2.5, fc='b',
                                    ls='dashdot', fill=False)

    ax.add_patch(patch_car)
    ax.add_patch(patch_goal)
    predict, = ax.plot([], [], 'r--', linewidth = 1)

    # Car steering and throttle position.
    telem = [3,14]
    patch_wheel = mpatches.Circle((telem[0]-3, telem[1]), 2.2)
    ax.add_patch(patch_wheel)
    wheel_1, = ax.plot([], [], 'k', linewidth = 3)
    wheel_2, = ax.plot([], [], 'k', linewidth = 3)
    wheel_3, = ax.plot([], [], 'k', linewidth = 3)
    throttle_outline, = ax.plot([telem[0], telem[0]], [telem[1]-2, telem[1]+2],
                                'b', linewidth = 20, alpha = 0.4)
    throttle, = ax.plot([], [], 'k', linewidth = 20)
    brake_outline, = ax.plot([telem[0]+3, telem[0]+3], [telem[1]-2, telem[1]+2],
                            'b', linewidth = 20, alpha = 0.2)
    brake, = ax.plot([], [], 'k', linewidth = 20)
    throttle_text = ax.text(telem[0], telem[1]-3, 'Forward', fontsize = 8,
                        horizontalalignment='center')
    brake_text = ax.text(telem[0]+3, telem[1]-3, 'Reverse', fontsize = 8,
                        horizontalalignment='center')

    # Obstacles
    if OBSTACLES:
        patch_obs = mpatches.Circle((mpc.x_obs, mpc.y_obs),0.5)
        patch_obs.set_facecolor("Orange")
        ax.add_patch(patch_obs)


    def steering_wheel(wheel_angle):
        wheel_1.set_data([telem[0]-3, telem[0]-3+np.cos(wheel_angle)*2],
                         [telem[1], telem[1]+np.sin(wheel_angle)*2])
        wheel_2.set_data([telem[0]-3, telem[0]-3-np.cos(wheel_angle)*2],
                         [telem[1], telem[1]-np.sin(wheel_angle)*2])
        wheel_3.set_data([telem[0]-3, telem[0]-3+np.sin(wheel_angle)*2],
                         [telem[1], telem[1]-np.cos(wheel_angle)*2])

    def update_plot(num):
        # Car.
        patch_car.set_xy(car_patch_pos(state_i[num,0], state_i[num,1], state_i[num,2]))
        patch_car.angle = np.rad2deg(state_i[num,2])-90
        # Car wheels
        np.rad2deg(state_i[num,2])
        steering_wheel(u_i[num,1]*2)
        throttle.set_data([telem[0],telem[0]],
                        [telem[1]-2, telem[1]-2+max(0,u_i[num,0]/5*4)])
        brake.set_data([telem[0]+3, telem[0]+3],
                        [telem[1]-2, telem[1]-2+max(0,-u_i[num,0]/5*4)])

        # Goal.
        if (num <= 100 or ref_2 == None):
            patch_goal.set_xy(car_patch_pos(ref_1[0],ref_1[1],ref_1[2]))
            patch_goal.angle = np.rad2deg(ref_1[2])-90
        else:
            patch_goal.set_xy(car_patch_pos(ref_2[0],ref_2[1],ref_2[2]))
            patch_goal.angle = np.rad2deg(ref_2[2])-90

        #print(str(state_i[num,3]))
        predict.set_data(predict_info[num][:,0],predict_info[num][:,1])
        # Timer.
        #time_text.set_text(str(100-t[num]))

        return patch_car, time_text


    # print("Compute Time: ", round(time.process_time() - start, 3), "seconds.")
    # Animation.
    car_ani = animation.FuncAnimation(fig, update_plot, frames=range(1,len(state_i)), interval=100, repeat=True, blit=False)
    # car_ani.save('mpc-video.mp4')
    return car_ani 


class ModelPredictiveControl:
    def __init__(self, horizon, goal1, goal2, obs):
        self.horizon = horizon
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = goal1
        self.reference2 = goal2

        if obs == None: 
            self.x_obs = -5
            self.y_obs = -5
        else: 
            self.x_obs = obs[0]
            self.y_obs = obs[1]

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]
        
        beta = steering
        a_t = pedal 
        
        x_dot = v_t * np.cos(psi_t)
        y_dot = v_t * np.sin(psi_t)
        psi_dot = v_t * np.tan(beta)/2.5 
        v_dot = a_t
        
        x_t += x_dot * dt 
        y_t += y_dot * dt 
        psi_t += psi_dot *dt
        v_t += v_dot * dt - v_t/25.0

        return [x_t, y_t, psi_t, v_t]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0 
        
        for k in range(0, self.horizon):
            ts = [0,1]
            v_start = state[3]
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1])
            cost += (ref[0] - state[0]) ** 2
            cost += (ref[1] - state[1]) ** 2 
            cost += (ref[2] - state[2]) ** 2 
            cost += self.obstacle_cost(state[0], state[1])
            
        return cost 
    
    def obstacle_cost(self, x, y):
        distance = (x - self.x_obs)**2 + (y - self.y_obs)**2 
        distance = np.sqrt(distance)
        # return 1/distance*100
        if (distance > 2): 
            return 10
        else: 
            return 1/distance*30
        
# Shift xy, centered on rear of car to rear left corner of car.
def car_patch_pos(x, y, psi, car_width=1.0):
    #return [x,y]
    x_new = x - np.sin(psi)*(car_width/2)
    y_new = y + np.cos(psi)*(car_width/2)
    return [x_new, y_new]    

def plot_car_obs_pos(goal1, goal2, obstacle):
    fig = plt.figure(figsize=[9,9])
    gs = gridspec.GridSpec(8,8)

    # Elevator plot settings.
    ax = fig.add_subplot(gs[:8, :8])

    plt.xlim(-3, 14)
    ax.set_ylim([-3, 14])
    plt.xticks(np.arange(0,11, step=2))
    plt.yticks(np.arange(0,11, step=2))
    plt.title('MPC 2D') 
    
    car_width = 1.0
    car_init_pos = mpatches.Rectangle((0,0), car_width, 2.5, fc='b',ls='dashdot', fill=True) 
    # car_init_pos.set_xy(car_patch_pos(0,0,0,car_width))
    car_init_pos.angle = np.rad2deg(0)-90 
    
    
    patch_goal1 = mpatches.Rectangle((goal1[0],goal1[1]), car_width, 2.5, fc='b',ls='dashdot', fill=False) 
    # patch_goal1.set_xy(car_patch_pos(goal1[0],goal1[1],goal1[2],car_width))
    patch_goal1.angle = np.rad2deg(goal1[2])-90 
    
    if goal2 != None:
        patch_goal2 = mpatches.Rectangle((goal2[0],goal2[1]), car_width, 2.5, fc='b',ls='dashdot', fill=False) 
        # patch_goal2.set_xy(car_patch_pos(goal2[0],goal2[1],goal2[2],car_width))
        patch_goal2.angle = np.rad2deg(goal2[2])-90 
    
    if obstacle!=None:
        patch_obs = mpatches.Circle(obstacle,0.5) 
        patch_obs.set_facecolor("Orange")
        ax.add_patch(patch_obs)

    ax.add_patch(car_init_pos)
    ax.add_patch(patch_goal1)
    if goal2 != None:
        ax.add_patch(patch_goal2)
    return fig

if __name__=="__main__":
    
    st.subheader("SIMULATION OF MODEL PREDICTIVE CONTROLLER (HIGHWAY)") 
    st.markdown("""
                MPC can be used to control any system, including system with contrains. 
                Here there is any object that should be avoided, this is done by setting a threshod
                and adding cost when it gets too close to the obstacle. 
                """)
    st.caption("Obstable : Orange Circle, dotted squares: Goal1 and Goal2, Blue square: car")
    st.divider()
    
    container1 = st.container()
    container2 = st.container()
    container3 = st.container()
    
    with container1:
        horizon = st.slider("Horizon: How far the controller must predict at each time step", min_value=1, max_value=35, value=10)
        with st.expander("Change Position of Goal and Obstacle"):
            col1_1, col2_1, col3_1 = st.columns(3) 
            with col1_1:        
                x1 = st.slider("x position reference 1", min_value=0, max_value=12, value=10)
                x2 = st.slider("x position reference 2", min_value=0, max_value=12, value=0)
            with col2_1:
                y1 = st.slider("y position reference 1", min_value=0, max_value=12, value=1)
                y2 = st.slider("y position reference 2", min_value=0, max_value=12, value=8)
            with col3_1:
                theta1 = st.slider("Angle reference 1", min_value=0, max_value=360, value=0) * (np.pi / 180)
                theta2 = st.slider("Angle reference 2", min_value=0, max_value=360, value=90)  * (np.pi / 180) 
            obs_yes = st.checkbox("Obstacle to avoid", value=True) 
            col1_2, col2_2 = st.columns(2)
            if obs_yes:
                with col1_2: 
                    obs_x = st.slider("x position obstacle", min_value=0, max_value=12, value=5) 
                with col2_2: 
                    obs_y = st.slider("y position obstacle", min_value=0.1, max_value=12.0, value=0.1) 
                obs_pos = [obs_x, obs_y]
            else: 
                obs_pos = None 
                
        goal1 = [x1, y1, theta1]
        goal2 = [x2, y2, theta2] 
        
        
    with container2:
        _,_,_,mid1,mid2,_,_ = st.columns(7)
        button = st.button
        run_toggle_flag = False 
        
        with mid1:
            if run_toggle_flag == False and button("Run"):
                run_toggle_flag = True
        with mid2:
            if run_toggle_flag == True and button("Stop"): 
                run_toggle_flag = False 
        # print(run_toggle_flag)
        
    if run_toggle_flag==False:
        with container3:
            figure = plot_car_obs_pos(goal1=goal1, goal2=goal2, obstacle=obs_pos)
            st.pyplot(figure)
    else: 
        with container3:
            start_time = time.process_time()
            animation = sim_run(options, ModelPredictiveControl, horizon, goal1, goal2, obs_pos) 
            end_time = time.process_time()
            # st.markdown("Simulation of Model Predictive Controller")
            animjs = animation.to_jshtml()
            
            ## JS line to find the play button and click on it
            click_on_play = """document.querySelector('.anim-buttons button[title="Play"]').click();""" 
            ## Search for the creation of the animation within the jshtml file created by matplotlib
            pattern = re.compile(r"(setTimeout.*?;)(.*?})", re.MULTILINE | re.DOTALL) 
            ## Insert the JS line right below that
            new_animjs = pattern.sub(rf"\1 \n {click_on_play} \2", animjs)
            
            
            components.html(new_animjs, height=700)
            
            st.markdown("Total Compute Time is {}".format(end_time-start_time))

