% 1. Domain Definition
#const max_row = 3.
#const max_col = 3.
#const max_time = 4. % t=0 to t=4, 5 timesteps total

row(1..max_row).
col(1..max_col).
time(0..max_time).
time_step(0..max_time-1). % Time steps during which actions are taken

robot(r1; r2).

% Action definitions: (ActionName, DeltaRow, DeltaCol)
action_type(wait, 0, 0).
action_type(up, -1, 0).
action_type(down, 1, 0).
action_type(left, 0, -1).
action_type(right, 0, 1).

% 2. Initial Positions at t=0
pos(r1, 1, 1, 0). % r1 starts at (1,1)
pos(r2, 3, 3, 0). % r2 starts at (3,3)

% Each robot is at exactly one position at t=0
1 { pos(R, X, Y, 0) : row(X), col(Y) } 1 :- robot(R).

% 3. Generate Actions for each robot at each time_step (t=0 to max_time-1)
% Each robot takes exactly one action at each time step
{ action(R, Type, DR, DC, T) : action_type(Type, DR, DC) } = 1 :- robot(R), time_step(T).

% 4. Effects: Determine position at T+1 based on action at T
% A robot's position at T+1 is determined by its position at T and the action it takes at T.
pos(R, R_next, C_next, T+1) :-
    pos(R, R_curr, C_curr, T),          % Current position at T
    action(R, _, DR, DC, T),            % Action taken at T
    R_next = R_curr + DR,               % Calculate new row
    C_next = C_curr + DC,               % Calculate new column
    time_step(T).                       % Only for T where actions are defined

% 5. Integrity Constraints

% a. Bounds Checking: Robots must stay within the grid
:- pos(R, Row, Col, T), not row(Row). % Row must be within bounds
:- pos(R, Row, Col, T), not col(Col). % Column must be within bounds

% b. No Vertex Collision: No two robots can occupy the same cell at the same time
:- pos(R1, Row, Col, T), pos(R2, Row, Col, T), R1 != R2.

% c. No Edge Collision: No two robots can swap adjacent cells in one step
% If R1 goes from P1 to P2 and R2 goes from P2 to P1 in the same time step.
:-  pos(R1, R1_curr, C1_curr, T),       % R1 is at (R1_curr, C1_curr) at T
    pos(R2, R2_curr, C2_curr, T),       % R2 is at (R2_curr, C2_curr) at T
    R1 != R2,                           % They are different robots
    pos(R1, R2_curr, C2_curr, T+1),     % R1 moves to R2's current position by T+1
    pos(R2, R1_curr, C1_curr, T+1),     % R2 moves to R1's current position by T+1
    time_step(T).                       % This applies for action steps

% d. Goal State: Both robots must be at their destinations at t=max_time
:- not pos(r1, 1, 3, max_time). % r1 needs to reach (1,3)
:- not pos(r2, 3, 1, max_time). % r2 needs to reach (3,1)

% 6. Show statement
#show pos/4.