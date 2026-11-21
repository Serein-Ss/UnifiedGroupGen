"""Ordering of endpoint orbits in teacher programs."""

ORBIT_ORDERS = ('random', 'site_symmetry')


def endpoint_orbit_order(endpoint, rng, strategy='random'):
    order = rng.permutation(len(endpoint.letters))
    if strategy == 'random':
        return order
    if strategy != 'site_symmetry':
        raise ValueError(f'Unknown orbit order: {strategy}')
    # Within one cell group |Stab| = |G| / multiplicity. Stable sorting
    # retains the random order within ties without relying on template names.
    return sorted(order, key=lambda i: endpoint.spec.template(endpoint.letters[i]).multiplicity)
