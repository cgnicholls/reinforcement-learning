import numpy as np

class Pong():
    
    def __init__(self, size):
        self.size = size
        self.is_playing = False
        self.paddle_x = self.size/2
        self.ball_x = self.size/2
        self.ball_y = 0
        self.ball_vx = 1
        self.ball_vy = 1

    def get_state(self):
        state = np.zeros([self.size,self.size])
        state[self.ball_y, self.ball_x] = 1
        state[self.size-1,self.paddle_x] = 1
        return state

    def reset(self):
        self.is_playing = True
        self.ball_x = self.size/2
        self.ball_y = 0
        self.ball_vx = np.random.choice([-1,1])
        self.ball_vy = 1
        return self.get_state()

    def step(self, a):
        reward = 0

        # Action 1 = move right, action 0 = move left
        if a == 1:
            if self.paddle_x < self.size-1:
                self.paddle_x += 1
        if a == 0:
            if self.paddle_x > 0:
                self.paddle_x -= 1

        # Update ball position and velocity
        # If the ball is at the left or right edge of the screen and about to
        # move off the screen, then reverse its x velocity
        if self.ball_x == 0:
            if self.ball_vx == -1:
                self.ball_vx = 1
        if self.ball_x == self.size-1:
            if self.ball_vx == 1:
                self.ball_vx = -1

        # If the ball is at the top of the screen, make it bounce back and give
        # a reward of 1.
        if self.ball_y == 0:
            self.ball_vy = 1
            reward = 1

        # If the ball is at the bottom of the screen we have lost
        if self.ball_y == self.size-1:
            reward = -1
            self.is_playing = False
            return self.get_state(), reward, True, None

        # If the ball is about to go off the screen, check that the paddle is
        # there. And if it is then reverse the y velocity
        if self.ball_y == self.size-2 and self.ball_vy == 1:
            if self.paddle_x == self.ball_x:
                self.ball_vy = -1

        # Update the ball's position
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Return the state, reward and that the frame was not terminal.
        return self.get_state(), reward, False, None

def test_pong(max_steps=100):
    env = Pong(10)
    obs = env.reset()
    print "First observation"
    print obs
    for t in range(max_steps):
        print "Enter action: 0 = left, 1 = right"
        a = int(raw_input())
        obs, reward, done, info = env.step(a)
        print obs
        print reward
        print done
