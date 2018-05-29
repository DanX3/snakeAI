from gym.envs.classic_control import rendering
import gym
import numpy as np
from time import sleep
from collections import deque


class SnakeEnv2(gym.Env):
    dir_map = {
        0: np.array((1, 0)),
        1: np.array((-1, 0)),
        2: np.array((0, 1)),
        3: np.array((0, -1))
    }

    def __init__(self, grid=np.array((10, 10)), render=True):
        self.__grid = grid
        self.__render = render
        self.__tile_size = 20
        self.viewer = None
        self.body = None

    def reset(self):
        if self.__render:
            resolution = (self.__tile_size * self.__grid)
            if self.viewer is None:
                self.viewer = rendering.Viewer(resolution[0], resolution[1])
            else:
                del self.viewer.geoms[:]
        # self.__direction = self.dir_map[2]
        updown = np.random.randint(2)
        self.__direction = np.array((updown, 0 if updown else 1)) * np.power(-1, np.random.randint(2))
        self.__changed_direction = True

        # body setup
        if self.body is not None:
            self.body.clear()
        head_pos = self.__grid/2
        head = self.__create_square(self.__grid/2, color=(.96, .26, .21))
        body1 = self.__create_square(head_pos + np.array((0, -1)))
        body2 = self.__create_square(head_pos + np.array((0, -2)))
        self.time_steps = 0
        self.body = deque([head, body1, body2])
        self.__fruit = self.__create_square(self.__new_fruit_pos(), color=(.29, .68, .31))
        if self.viewer is not None:
            self.viewer.add_geom(head)
            self.viewer.add_geom(body1)
            self.viewer.add_geom(body2)
            self.viewer.add_geom(self.__fruit)

        return self.get_state()


    def render(self, mode='human'):
        if self.viewer is not None:
            return self.viewer.render()

    def step(self, action):
        self.time_steps += 1

        # Direction change with flag buffering for optimization - - - - - - - -
        new_dir = self.__new_dir(action)
        if (new_dir == self.__direction).all():
            self.__changed_direction = False
        else:
            self.__direction = new_dir
            self.__changed_direction = True

        # Head move - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        head_pos = self.get_geom_pos(self.body[0]) / self.__tile_size
        new_head_pos = (self.get_geom_pos(self.body[0]) + self.__direction * self.__tile_size) / self.__tile_size

        # quitting conditions - - - - - - - - - - - - - - - - - - - - - - - - -
        if not self.__validate_pos(new_head_pos, 1, len(self.body)-1) \
                or self.time_steps >= 400:
            return self.get_state(), 0.0, True, {'len': len(self.body), 't': self.time_steps}

        # check if fruit is eaten - - - - - - - - - - - - - - - - - - - - - - -
        self.set_geom_pos(self.body[0], new_head_pos * self.__tile_size)
        if (self.get_geom_pos(self.body[0]) == self.get_geom_pos(self.__fruit)).all():
            new_body_block = self.__create_square(head_pos)
            if self.viewer is not None:
                self.viewer.add_geom(new_body_block)
            head = self.body.popleft()
            self.body.appendleft(new_body_block)
            self.body.appendleft(head)
            self.set_geom_pos(self.__fruit, self.__new_fruit_pos() * self.__tile_size)
        else:
            head = self.body.popleft()
            self.body.appendleft(self.body.pop())
            self.body.appendleft(head)
            self.set_geom_pos(self.body[1], head_pos * self.__tile_size)

        return self.get_state(), 0.0, False, {'len': len(self.body), 't': self.time_steps}

    def __new_fruit_pos(self):
        while True:
            new_fruit_pos = self.__get_rand_tile()
            if self.__validate_pos(new_fruit_pos, 0, len(self.body)):
                break
        return new_fruit_pos

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def test(self):
        # test 1
        for i in range(50):
            rand_tile = self.__get_rand_tile()
            assert rand_tile[0] >= 0
            assert rand_tile[0] <= self.__grid[0]
            assert rand_tile[1] >= 0
            assert rand_tile[1] <= self.__grid[1]
        print "Test #1 passed (rand_tile)"

        # test 2
        assert self.pos_in_body(np.array((5, 5)), 0, len(self.body))
        assert self.pos_in_body(np.array((5, 4)), 0, len(self.body))
        assert not self.pos_in_body(np.array((1, 2)), 0, len(self.body))
        assert not self.pos_in_body(np.array((2, 1)), 0, len(self.body))
        print "Test #2 passed (__array_in_list)"

        # test 3
        self.__direction = self.dir_map[0]
        assert (self.__direction == self.__new_dir(0)).all()
        assert (self.__direction == self.__new_dir(1)).all()
        assert not (self.__direction == self.__new_dir(2)).all()
        assert not (self.__direction == self.__new_dir(3)).all()
        print "Test #3 passed (__new_dir)"

        # test 4
        assert self.__validate_pos(np.array((0, 0)), 0, len(self.body))
        assert self.__validate_pos(np.array((0, 1)), 0, len(self.body))
        assert self.__validate_pos(np.array((1, 0)), 0, len(self.body))
        assert not self.__validate_pos(np.array((0, -1)), 0, len(self.body))

        # test 5
        print self.__tiles_to_obs()
        # assert self.__tiles_to_obs() == (self.__grid/2)[1] - 1
        if False:
            print self.__fruit
            print self.body[0]
            print 'tiles to fruit', self.__tiles_to_fruit()
            print 'tiles to obs', self.__tiles_to_obs()

    def __get_rand_tile(self):
        x = np.random.randint(0, self.__grid[0])
        y = np.random.randint(0, self.__grid[1])
        return np.array((x, y))

    def __create_square(self, pos, color=(.2, .2, .2)):
        s = self.__tile_size
        geom = rendering.FilledPolygon([(0, 0), (0, s), (s, s), (s, 0)])
        r, g, b = color
        geom.set_color(r, g, b)
        transform = rendering.Transform(translation=pos*self.__tile_size)
        geom.add_attr(transform)
        return geom


    def __tiles_to_fruit(self):
        head_pos = np.array(self.get_geom_pos(self.body[0])) / self.__tile_size
        fruit_pos = np.array(self.get_geom_pos(self.__fruit))/ self.__tile_size
        return np.sum(np.abs(head_pos - fruit_pos))

    def __tiles_to_obs(self):
        """
        Computes the tiles that are between the head of the snake and an
        obstacle. An obstacle can be it's own body or the end of the grid
        TODO: caching to optimize performance but the body moves and the amount
        of tiles can change even if the direction does not. Fix this
        """
        if not self.__changed_direction:
            self.__prev_obs -= 1
            return self.__prev_obs

        cursor = self.get_geom_pos(self.body[0]) / self.__tile_size
        distance = -1
        while self.__validate_pos(cursor, 1, len(self.body)-1):
            distance += 1
            cursor += self.__direction
        self.__prev_obs = distance
        return distance


    def pos_in_body(self, a, body_start, body_end):
        for x in range(body_start, body_end):
            coords = self.get_geom_pos(self.body[x]) / self.__tile_size
            if (a == coords).all():
                return True
        return False

    def __new_dir(self, action):
        new_dir = self.dir_map[action]
        if (np.abs(new_dir) != np.abs(self.__direction)).all():
            return new_dir
        else:
            return self.__direction

    def get_state(self):
        return self.get_geom_pos(self.body[0]), \
               self.get_geom_pos(self.__fruit), \
               self.__direction, \
               self.__tiles_to_obs(), \
               self.__tiles_to_fruit()

    def get_geom_pos(self, geom):
        return np.array(geom.attrs[1].translation)

    def set_geom_pos(self, geom, pos):
        geom.attrs[1].translation = (pos[0], pos[1])

    def __validate_pos(self, pos, body_start, body_end):
        if (pos < self.__grid).all() \
                and (pos >= np.array((0, 0))).all() \
                and not self.pos_in_body(pos, body_start, body_end):
            return True
        return False



if __name__ == "__main__":
    env = SnakeEnv2()
    env.reset()
    env.test()
    env.reset()
    env.render()
    dir_map = { 'd': 0, 'a': 1, 'w': 2, 's': 3 }
    while True:
        direction = raw_input()
        s, r, d, o = env.step(dir_map[direction])
        env.render()
        # sleep(1)
        if d:
            break
    env.close()
