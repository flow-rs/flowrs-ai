
#[cfg(test)]
mod nodes {
    use std::{sync::mpsc::channel, thread, time::Duration};

    use flowrs::{
        connection::connect,
        exec::{
            execution::{Executor, StandardExecutor},
            node_updater::{MultiThreadedNodeUpdater, NodeUpdater, SingleThreadedNodeUpdater},
        },
        flow_impl::Flow,
        node::ChangeObserver,
        sched::round_robin::RoundRobinScheduler,
    };

    use flowrs_ai::{cluster::ClusterNode, scale::ScaleNode, csv_to_array::CSVToArrayNNode, dimred::DimRedNode};
    use flowrs_std::{
        debug::DebugNode,
        timer::{PollTimer, TimerStrategy, WaitTimer},
        value::ValueNode,
    };
    use serde::{Deserialize, Serialize};
    use ndarray::{Array2, arr2};

    fn connect_test_with<
        T: TimerStrategy<bool> + Send + 'static,
        U: NodeUpdater + Drop + Send + 'static,
    >(
        node_updater: U,
        timer: T,
    ) where
        T: Clone + Deserialize<'static> + Serialize,
    {
        let sleep_seconds = 1;
        let timer_interval_seconds = 1;

        let change_observer: ChangeObserver = ChangeObserver::new();
        let (sender, receiver) = channel::<bool>();

        let node_1 = ValueNode::new(
            arr2(&[[1, 2], [3, 4], [5, 6]]),
            Some(&change_observer),
        );

        let node_2: CSVToArrayNNode<> = CSVToArrayNNode::new(Some(&change_observer));
        let node_3: ScaleNode<> = ScaleNode::new(Some(&change_observer));
        let node_4: DimRedNode<> = DimRedNode::new(Some(&change_observer));
        let node_5: ClusterNode<> = ClusterNode::new(Some(&change_observer));

        let node_6 = DebugNode::<Array2<u8>>::new(Some(&change_observer));


        connect(node_1.output.clone(), node_2.input.clone());
        connect(node_2.output.clone(), node_3.input.clone());
        connect(node_3.output.clone(), node_4.input.clone());
        connect(node_4.output.clone(), node_5.input.clone());
        connect(node_5.output.clone(), node_6.input.clone());

        let mut flow: Flow = Flow::new_empty();
        flow.add_node(node_1);
        flow.add_node(node_2);
        flow.add_node(node_3);
        flow.add_node(node_4);
        flow.add_node(node_5);
        flow.add_node(node_6);

        let (controller_sender, controller_receiver) = channel();
        let thread_handle = thread::spawn(move || {
            let mut executor = StandardExecutor::new(change_observer);

            controller_sender
                .send(executor.controller())
                .expect("Controller sender cannot send.");

            executor
                .run(flow, RoundRobinScheduler::new(), node_updater)
                .expect("Run failed.");
        });

        let controller = controller_receiver.recv().unwrap();

        thread::sleep(Duration::from_secs(sleep_seconds));

        //println!("                                      ----> {:?} CANCEL", std::thread::current().id());

        controller.lock().unwrap().cancel();

        thread_handle.join().unwrap();

        let num_iters = receiver.iter().count();

        let asserted_num_iters = sleep_seconds / timer_interval_seconds;

        //println!("{} {}", num_iters, asserted_num_iters.abs_diff(num_iters as u64));
        assert!(asserted_num_iters.abs_diff(num_iters as u64) <= 1);
    }

    #[test]
    fn test() {
        connect_test_with(MultiThreadedNodeUpdater::new(4), WaitTimer::new(true));

    }
}
