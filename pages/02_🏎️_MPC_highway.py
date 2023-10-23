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
options['FULL_RECALCULATE'] = False

def sim_run(options, MPC, horizon, reference, limit):
    start = time.process_time()
    # Simulator Options
    FIG_SIZE = options['FIG_SIZE'] # [Width, Height]
    FULL_RECALCULATE = options['FULL_RECALCULATE']

    mpc = MPC(horizon, reference, limit)

    num_inputs = 2
    u = np.zeros(mpc.horizon*num_inputs)
    bounds = []

    # Set bounds for inputs bounded optimization.
    for i in range(mpc.horizon):
        bounds += [[-1, 1]]
        bounds += [[-0.001, 0.001]]

    ref = mpc.reference

    state_i = np.array([[1,0,0,0]])
    u_i = np.array([[0,0]])
    sim_total = 200
    predict_info = [state_i]

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for i in range(1,sim_total+1):
        # Reuse old inputs as starting point to decrease run time.
        u = np.delete(u,0)
        u = np.delete(u,0)
        u = np.append(u, u[-2])
        u = np.append(u, u[-2])
        if (FULL_RECALCULATE):
            u = np.zeros(mpc.horizon*num_inputs)
        start_time = time.time()

        # Non-linear optimization.
        u_solution = minimize(mpc.cost_function, u, (state_i[-1], ref),
                                method='SLSQP',
                                bounds=bounds,
                                tol = 1e-5)
        # print('Step ' + str(i) + ' of ' + str(sim_total) + '   Time ' + str(round(time.time() - start_time,5)))
        u = u_solution.x
        y = mpc.plant_model(state_i[-1], mpc.dt, u[0], u[1])
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


    ax.set_ylim(-3, 17)
    plt.xticks(np.arange(0,90, step=2))
    plt.xlim(0, 20)
    plt.yticks([])
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
    telem = [6,14]
    patch_wheel = mpatches.Circle((telem[0]-3, telem[1]), 2.2)
    ax.add_patch(patch_wheel)
    wheel_1, = ax.plot([], [], 'k', linewidth = 3)
    wheel_2, = ax.plot([], [], 'k', linewidth = 3)
    wheel_3, = ax.plot([], [], 'k', linewidth = 3)
    #throttle_outline, = ax.plot([telem[0], telem[0]], [telem[1]-2, telem[1]+2],
    #                            'b', linewidth = 20, alpha = 0.4)
    throttle, = ax.plot([], [], 'k', linewidth = 20)
    #brake_outline, = ax.plot([telem[0]+3, telem[0]+3], [telem[1]-2, telem[1]+2],
    #                        'b', linewidth = 20, alpha = 0.2)
    brake, = ax.plot([], [], 'k', linewidth = 20)
    throttle_text = ax.text(telem[0], telem[1]-3, 'Forward', fontsize = 15,
                        horizontalalignment='center')
    brake_text = ax.text(telem[0]+3, telem[1]-3, 'Reverse', fontsize = 15,
                        horizontalalignment='center')


    # Speed Indicator
    speed_text = ax.text(telem[0]+7, telem[1], '0', fontsize=15)
    speed_units_text = ax.text(telem[0]+8.5, telem[1], 'km/h', fontsize=15)
    
    # Speed Limit Indicator
    limit_box = mpatches.FancyBboxPatch([telem[0]+7, telem[1]+1],0.9,0.5)
    limit_box.set_facecolor("Red")
    speed_limit = ax.text(telem[0]+7, telem[1]+1, limit, fontsize=15,)
    speed_limit.set_color('White') 
    speed_limit_text = ax.text(telem[0]+8.5, telem[1]+1, 'limit', fontsize=15)
    ax.add_patch(limit_box)
    # Shift xy, centered on rear of car to rear left corner of car.
    def car_patch_pos(x, y, psi):
        x_new = x - np.sin(psi)*(car_width/2)
        y_new = y + np.cos(psi)*(car_width/2)
        return [x_new, y_new]

    def steering_wheel(wheel_angle):
        wheel_1.set_data([telem[0]-3, telem[0]-3+np.cos(wheel_angle)*2],
                         [telem[1], telem[1]+np.sin(wheel_angle)*2])
        wheel_2.set_data([telem[0]-3, telem[0]-3-np.cos(wheel_angle)*2],
                         [telem[1], telem[1]-np.sin(wheel_angle)*2])
        wheel_3.set_data([telem[0]-3, telem[0]-3+np.sin(wheel_angle)*2],
                         [telem[1], telem[1]-np.cos(wheel_angle)*2])
        brake_text.set_x(telem[0]+3)
        throttle_text.set_x(telem[0])
        patch_wheel.center = telem[0]-3, telem[1]
        speed_text.set_x(telem[0]+7)
        speed_units_text.set_x(telem[0]+8.5)

    def update_plot(num):
        # Car.
        patch_car.set_xy(car_patch_pos(state_i[num,0], state_i[num,1], state_i[num,2]))
        patch_car.angle = np.rad2deg(state_i[num,2])-90
        # Car wheels
        steering_wheel(u_i[num,1]*2)
        throttle.set_data([telem[0],telem[0]],
                        [telem[1]-2, telem[1]-2+max(0,u_i[num,0]/1*4)])
        brake.set_data([telem[0]+3, telem[0]+3],
                        [telem[1]-2, telem[1]-2+max(0,-u_i[num,0]/1*4)])

        speed = state_i[num,3]*3.6
        speed_text.set_text(str(round(speed,1)))
        if speed > limit:
            speed_text.set_color('r')
        else:
            speed_text.set_color('k')
        
        limit_box.set_x(telem[0]+7) 
        limit_box.set_y(telem[1]+1)
        
        speed_limit_text.set_x(telem[0]+8.5)
        speed_limit.set_x(telem[0]+7)
                
        patch_goal.set_xy(car_patch_pos(ref[0],ref[1],ref[2]))
        patch_goal.angle = np.rad2deg(ref[2])-90



        #print(str(state_i[num,3]))
        predict.set_data(predict_info[num][:,0],predict_info[num][:,1])
        # Timer.
        #time_text.set_text(str(100-t[num]))
        if (state_i[num,0] > 5):
            plt.xlim(state_i[num,0]-5, state_i[num,0]+15)
            telem[0] = state_i[num,0]+1


        return patch_car, time_text


    # print("Compute Time: ", round(time.process_time() - start, 3), "seconds.")
    # Animation.
    car_ani = animation.FuncAnimation(fig, update_plot, frames=range(1,len(state_i)), interval=100, repeat=True, blit=False)
    #car_ani.save('mpc-video.mp4')
    return car_ani 
    # plt.show()


class ModelPredictiveControl:
    def __init__(self, horizon, reference, speed_limit):
        self.horizon = horizon # 20
        self.dt = 0.2
        self.speed_limit = speed_limit

        # Reference or set point the controller will achieve.
        self.reference = reference # [50, 0, 0]

    def plant_model(self, prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        v_t = prev_state[3] # m/s
        a_t = pedal 
        
        x_t_1 = x_t + v_t * dt
        v_t_1 = v_t + a_t * dt - v_t/25
        # v_t_1 = v_t * 0.96
        return [x_t_1, 0, 0, v_t_1]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0
        
        for k in range(0, self.horizon):
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1]) 
            cost += (ref[0]-state[0])**2

            speed_kph = state[3] * 3.6 # m/s to km/h
            if speed_kph > self.speed_limit:
                cost += abs(speed_kph-self.speed_limit)*100
        
        return cost


def plot_car_goal_pos(goal):
    fig = plt.figure(figsize=[9,9])
    gs = gridspec.GridSpec(8,8)
    
    # Elevator plot settings.
    ax = fig.add_subplot(gs[:8, :8]) 
    
    plt.xlim(-1, goal[0]+5)
    ax.set_ylim([-goal[0]//4, (goal[0]*3)//4])
    plt.xticks(np.arange(-1, goal[0]+5, step=4 if goal[0]<20 else 10))
    # plt.yticks(np.arange(-goal[0]//4, (goal[0]*3)//4, step=4 if goal[0]<20 else 10))
    
    # plt.title('MPC 2D')
    
    car_width = 1.0
    car_init_pos = mpatches.Rectangle((0,0), car_width, 2.5, fc='b',ls='dashdot', fill=True) 
    car_init_pos.angle = np.rad2deg(0)-90 
    
    patch_goal = mpatches.Rectangle((goal[0],goal[1]), car_width, 2.5, fc='b',ls='dashdot', fill=False)
    patch_goal.angle = np.rad2deg(goal[2])-90 
    
    ax.add_patch(car_init_pos)
    ax.add_patch(patch_goal)
    ax.get_yaxis().set_visible(False)
    
    return fig 

if __name__=="__main__":
    toggle_flag = False 
    
    st.subheader("SIMULATION OF MODEL PREDICTIVE CONTROLLER (HIGHWAY)") 
    st.markdown("""
                MPC uses the model of a system to predict its future behavior, 
                and it solves an optimization problem to select the best control action. 
                Since I love cars🚗, I obiously implemented MPC to control it. 😅
                """)
    st.divider()
    container1 = st.container()
    container2 = st.container() 
    container3 = st.container() 

    
    with container1:
        horizon = st.slider("Horizon: How far the controller must predict at each time step", min_value=1, max_value=35, value= 10)
        with st.expander("Change parameters"):
            goal_x = st.slider("x position of goal location", min_value=5, max_value=80, value=50)
            speed_limit = st.slider("Speed limit", min_value=5, max_value=20, value=15) 
    
    goal = [goal_x, 0, 0]
    
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
            figure = plot_car_goal_pos(goal) 
            st.pyplot(figure)
    else: 
        with container3:
            start_time = time.process_time()
            animation = sim_run(options, ModelPredictiveControl, horizon, goal, speed_limit)
            end_time = time.process_time()
            st.subheader("Simulation of Model Predictive Controller on a highway")
            animjs = animation.to_jshtml()
            
            ## JS line to find the play button and click on it
            click_on_play = """document.querySelector('.anim-buttons button[title="Play"]').click();""" 
            ## Search for the creation of the animation within the jshtml file created by matplotlib
            pattern = re.compile(r"(setTimeout.*?;)(.*?})", re.MULTILINE | re.DOTALL) 
            ## Insert the JS line right below that
            new_animjs = pattern.sub(rf"\1 \n {click_on_play} \2", animjs)
            
            
            components.html(new_animjs, height=700) # height=800  
            
            st.markdown("Total Compute Time is {}".format(end_time-start_time))