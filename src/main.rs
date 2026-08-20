use network_sim::{
    network::{
        SingleLink,
        SingleLinkRoute,
        alloc::MaxMin,
    },
    Simulator,
};

fn main() {
    // Create a single link topology with 1 Gbps capacity and shortest path routing
    let single_link_router = SingleLinkRoute::new();
    let single_link = SingleLink::new(1.0e9, single_link_router);

    let mut sim = Simulator::new(single_link, MaxMin::new());
    sim.add_flow_arrival(0, 0, 1, 50 * (1 << 20), 0);   // 50 MiB, job_flow_idx=0
    sim.add_flow_arrival(500, 0, 1, 30 * (1 << 20), 1); // 30 MiB at 0.5 s, job_flow_idx=1

    while let Some(kind) = sim.advance_next_step() {
        println!("{kind:?} @ {} us – rates: {:?}", sim.now_us, sim.get_rates());
    }
}
