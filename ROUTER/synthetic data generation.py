import json
import pandas as pd
from google import genai
from google.genai import types

YOUR_API_KEY = ''
MODEL_ID ="gemini-2.0-flash-001"
client = genai.Client(api_key=YOUR_API_KEY)

document_keywords = """
aTokens (V2): Interest-bearing tokens representing deposits (with EIP-2612 support).
Debt Tokenization: Representing borrows as specific ERC20 tokens (Variable Debt Tokens, Stable Debt Tokens).
Scaled Balance (ScB): Efficient internal accounting method for aToken balances.
Scaled Variable Debt (ScVB): Efficient internal accounting for variable debt token balances.
Variable Rate Borrowing: Borrowing at an interest rate that fluctuates with market conditions.
Stable Rate Borrowing (V2): Borrowing at a rate fixed at the time of the loan, allowing multiple stable positions per asset.
Flash Loans V2: Uncollateralized loans (repaid in same transaction) with enhanced composability (usable within Aave actions) via a pull repayment mechanism.
Credit Delegation: Delegating borrowing power to another address without collateral transfer (via approveDelegation).
Collateral Swap/Trading: Using Flash Loans V2 to exchange deposited collateral assets directly.
Repay with Collateral: Using Flash Loans V2 to repay debt using deposited collateral.
LendingPool (V2 Architecture): Central contract for all user interactions.
Library-Based Architecture: Replacing core logic contracts with libraries for gas savings and modularity.
User Reserve Bitmask: Compact (256-bit) representation tracking user deposits/borrows across assets (up to 128).
Reserve Configuration Bitmask: Compact storage for asset parameters (LTV, threshold, flags).
pow Function Approximation: Gas optimization using binomial expansion for interest calculation.
SafeMath Removal (Optimization): Removing SafeMath checks in specific math libraries for gas efficiency.
Balancer Pool: An automated market maker functioning as a self-balancing weighted portfolio.
Value Function (V): Core mathematical invariant (V = ∏ B_t ^ W_t) defining the trading curve.
Weighted Portfolio: Portfolio where assets are held according to predefined target weights.
Self-Balancing: Portfolio automatically rebalances through arbitrage trading incentives.
Spot Price (SP): Theoretical instantaneous price between two assets based on balances and weights.
Effective Price (EP): Actual average price paid for a trade of a given size, accounting for slippage.
Constant Value Distribution: Property where the value share of each asset remains proportional to its weight.
Pool Tokens (Balancer): Tokens representing proportional ownership (liquidity provision) in a pool.
Liquidity Provision (LP): Supplying assets to a pool to earn trading fees.
Trading Fees (Balancer): Fees paid by traders, collected by liquidity providers.
Arbitrageurs: Traders who exploit price differences, thereby rebalancing the pool.
All-Asset Deposit/Withdrawal: Adding/removing liquidity proportionally across all assets in the pool.
Single-Asset Deposit/Withdrawal: Adding/removing liquidity using only one specific asset in the pool (incurs implicit trading/fees).
Controlled Pools: Pools with a designated "controller" address that can change parameters (weights, assets, fees).
Finalized Pools: Pools with fixed parameters (assets, weights, fees), enabling trustless public deposits/withdrawals.
Swap Fees: Fees charged on token exchanges within the pool.
Exit Fees: Fees charged when liquidity providers remove liquidity (partially returned to remaining LPs).
Peer-to-Peer Electronic Cash: Direct online payments without a financial institution intermediary.
Chain of Digital Signatures: Defining coin ownership through successive signed transfers.
Double-Spending Problem: Risk of the same digital coin being spent more than once.
Proof-of-Work (PoW): Consensus mechanism requiring computational effort (CPU power) to validate transactions and create blocks (based on Hashcash).
Timestamp Server (Distributed): P2P network service ordering transactions chronologically using PoW.
Blockchain: Chain of blocks, where each block timestamps transactions and references the previous block's hash.
Hashing (SHA-256): Cryptographic function used for proof-of-work, block linking, and transaction integrity (Merkle Trees).
Nonce: Value incremented in the block header during PoW mining to find a valid hash.
Network Nodes: Participants running the Bitcoin software, validating and relaying transactions/blocks.
Longest Chain Rule: Consensus rule where nodes accept the valid chain with the most cumulative proof-of-work.
Block Reward: Incentive (newly created coins + fees) for nodes successfully mining a block.
Transaction Fees: Optional fees paid by senders to incentivize miners to include transactions.
Merkle Tree: Data structure used to efficiently summarize transactions within a block hash.
Reclaiming Disk Space: Pruning old, spent transactions from the Merkle Tree to save storage.
Simplified Payment Verification (SPV): Verifying transactions using only block headers and Merkle branches, without a full node.
Transaction Inputs/Outputs (UTXO Model): Model where transactions consume previous outputs and create new ones (unspent transaction outputs).
Privacy (via Anonymous Keys): Maintaining privacy by not linking public keys to real-world identities.
51% Attack: Potential attack where a single entity controlling majority CPU power could reverse transactions or prevent confirmations.
Decentralized Oracle Networks (DONs): The foundational concept - committees of nodes providing diverse oracle services beyond data feeds.
Hybrid Smart Contracts: Secure composition of on-chain (smart contract) and off-chain (DON) components.
Executables (DON): Deterministic code/programs running continuously on a DON.
Adapters (DON): Interfaces connecting DONs to external resources (blockchains, APIs, storage, other DONs), potentially bidirectional.
Initiators (DON): Event-driven triggers for Executable logic (generalization of Keepers).
Ledger (DON): The underlying, ordered data structure maintained by a DON (often via BFT consensus).
Decentralized Metalayer: The long-term vision of an abstraction layer simplifying development across on-chain/off-chain systems.
Fair Sequencing Services (FSS): A DON service providing fair transaction ordering to combat front-running/MEV.
Proof of Reserves: Using DONs to attest to collateral backing assets across chains.
Decentralized Identity (DID) (via DONs): Using DONs with tools like DECO/Town Crier for privacy-preserving credential issuance and management.
Confidentiality-Preserving DeFi / Mixicles: Financial instruments enabled by DONs that conceal transaction details or underlying assets.
Priority Channels: A DON service guaranteeing timely inclusion of critical transactions on a blockchain.
Interfacing with Enterprise/Legacy Systems: Using DONs as secure middleware.
Transaction-Execution Framework (TEF): A framework for integrating DONs with Layer-2 scaling solutions (e.g., rollups) for performant hybrid contracts.
Anchor Contract (TEF): The on-chain smart contract component in the TEF, managing assets and verifying state updates.
Off-Chain Reporting (OCR): An existing BFT-like protocol for efficient report aggregation, a stepping stone to DONs.
DECO (Decentralized Oracle): Cryptographic protocol for proving data from TLS sessions without revealing sensitive info (no TEE needed).
Town Crier: Oracle technology using Trusted Execution Environments (TEEs) for data integrity and confidentiality.
Verifiable Random Functions (VRF): Cryptographic primitive providing provably fair randomness.
Trusted Execution Environments (TEEs): Secure hardware enclaves for confidential computation.
Secure Multi-Party Computation (MPC): Cryptographic techniques for computing on secret-shared data.
Functional Signatures (& Discretized): Cryptographic signatures proving a computation over signed inputs, useful for authenticated data aggregation.
Authenticated Data Origination (ADO): Enabling data sources to directly sign and authenticate their API data via Chainlink nodes.
Public-Key Infrastructure (PKI) (Chainlink): System for binding identities (via ENS like data.eth) to keys within the ecosystem.
Trust Minimization: Design goal to reduce reliance on the honesty of individual components.
Order-Fairness: Policy ensuring transactions are ordered fairly (e.g., temporally).
Secure Causality Preservation: Cryptographic techniques (commit-reveal, threshold encryption) to hide transaction content until ordering is fixed.
Front-Running / MEV / REV: Malicious strategies exploiting transaction ordering, addressed by FSS.
Minority Reports: Mechanism allowing a minority of DON nodes to flag potential majority malfeasance.
Guard Rails: On-chain safety mechanisms (e.g., circuit breakers) in smart contracts to protect against DON failures.
Failover Clients: Maintaining backup client versions for resilience against software exploits.
Staking (Chainlink 2.0): Depositing LINK as collateral, subject to slashing for misbehavior, designed for report correctness.
Slashing: Confiscation of staked assets due to misbehavior.
Super-linear Staking Impact (Quadratic): Economic property where the cost to bribe the network grows faster (e.g., quadratically) than the total stake.
Future Fee Opportunity (FFO): Implicit incentive derived from potential future earnings for reputable node operators.
Implicit-Incentive Framework (IIF): System for measuring and incorporating implicit incentives like FFO into economic security calculations.
Prospective Bribery: Advanced adversarial strategy offering bribes contingent on future random selection/outcomes, considered in the staking design.
Watchdog Priority: Mechanism in the staking design assigning priorities for alerting, concentrating rewards to deter bribery.
Two-Tier Adjudication: Staking system using a primary DON tier and a high-trust second tier for dispute resolution.
Virtuous Cycle of Economic Security: Positive feedback loop where network usage and FFO drive staking and security, lowering costs and attracting more users.
Compound III: The name of the protocol version.
Comet: The primary smart contract instance for a specific market (e.g., cUSDCv3).
Base Asset: The single borrowable asset in a Comet market (e.g., USDC).
Collateral Asset: Assets supplied by users to enable borrowing the base asset.
EVM Compatible: Runs on Ethereum Virtual Machine chains.
Proxy Contract (cUSDCv3): The fixed address used for interacting with a market, enabling upgrades.
Implementation Contract: The logic contract behind the proxy.
Extension Contract (Ext): Auxiliary contract for additional, non-core features.
Configurator: Contract used by governance to set market parameters and deploy implementations.
Comet Factory: Contract used by the Configurator to deploy implementation instances.
Proxy Admin: Administrative contract controlling proxy upgrades (typically OpenZeppelin's).
Interest Rate Model: Determines supply and borrow rates based on utilization.
Utilization Rate: The ratio of borrowed base asset to supplied base asset.
Kink (Interest Rate Model): The utilization point where the interest rate slope changes.
Reserves (Base/Collateral): Protocol-owned funds protecting against bad debt.
Target Reserves: Governance-set reserve level influencing collateral purchasing.
Supply (Collateral/Base): Depositing assets into the protocol.
Withdraw (Collateral): Removing collateral assets.
Borrow (Base Asset): Borrowing the base asset (using the withdraw function).
Repay (Base Asset): Repaying borrowed base asset (using the supply function).
Liquidation: Process of seizing collateral from underwater accounts.
Absorb: Function called by liquidators to seize collateral and repay debt of an underwater account.
Buy Collateral: Function allowing users to purchase seized collateral from the protocol at a discount.
Allow: Granting permission for a manager address to act on an account's behalf.
Transfer (Base/Asset): Moving assets between accounts within the protocol.
Claim Rewards: Withdrawing accrued reward tokens.
Supply Rate: Interest rate earned by suppliers of the base asset.
Borrow Rate: Interest rate paid by borrowers of the base asset.
Borrow Collateral Factor: Determines initial borrowing capacity based on collateral value.
Liquidation Collateral Factor: Determines the threshold at which an account becomes liquidatable.
Base Borrow Min: Minimum required size for an initial borrow position.
Ask Price: Discounted price at which the protocol sells seized collateral.
StoreFrontPriceFactor: Governance parameter affecting the collateral discount.
Principal Value: Internal representation of balance before interest accrual.
Supply/Borrow Index: Global value tracking cumulative interest.
Account Management: Allowing managers to act on behalf of owners.
Allow By Signature (EIP-712): Gasless permission granting via offline signature.
User Nonce: Account-specific nonce for EIP-712 signatures.
Liquidatable Account (isLiquidatable): An account eligible for liquidation.
Borrow Collateralization (isBorrowCollateralized): Check if an account has sufficient collateral to borrow more.
Liquidator Points: Data structure tracking liquidator activity and gas spent.
Audits (OpenZeppelin, ChainSecurity): Mention of third-party security reviews.
Protocol Rewards (COMP, etc.): Incentive distribution mechanism.
Rewards Contract: External contract managing reward distribution and claims.
Reward Accrual Tracking: Internal accounting of earned rewards.
Governance (COMP-based): Protocol control via token holders/delegates.
Timelock: Contract enforcing execution delay for governance proposals.
Multi-chain Governance: Framework for managing deployments across different chains.
Bridge Receiver: Contract facilitating cross-chain governance communication.
Bulker Contract: Helper for executing multiple actions in one transaction.
Invoke: Function on the Bulker contract to execute batched actions.
Action Codes (bytes32): Constants identifying actions for the Bulker.
ERC-20 Compatibility: The Comet contract's interface for the base asset.
Cosmos: The overall network of interoperable blockchains.
Network of Distributed Ledgers: The fundamental description of Cosmos.
Cosmos Hub: The first blockchain in the network, acting as a central ledger for inter-zone tokens.
Zones: Independent blockchains connected to the Hub.
Tendermint Core: The underlying consensus engine software.
Tendermint BFT Consensus: The specific Byzantine Fault Tolerant consensus algorithm used (PBFT-like).
Proof-of-Stake (PoS): The mechanism securing the Hub and potentially Zones.
Atom Token: The native staking token of the Cosmos Hub.
Validators: Nodes with voting power participating in consensus.
Voting Power: The weight assigned to a validator in the consensus process (often based on staked Atoms).
Super-majority (>⅔): The threshold required for BFT consensus decisions (e.g., PreCommits).
Locking Mechanism: Consensus rule preventing validators from equivocation.
Fork Accountability: Guarantees that validators causing forks can be identified and punished.
PreVote / PreCommit: Stages within the Tendermint consensus voting process.
Polka: A collection of >⅔ PreVotes for a block in a round.
Commit: A collection of >⅔ PreCommits for a block in a round.
Inter-blockchain Communication (IBC): The protocol enabling communication and token transfers between zones and the Hub.
IBCBlockCommitTx: IBC transaction type proving a block commit from one chain to another.
IBCPacketTx: IBC transaction type proving a specific packet was published on a source chain.
Coin Packet: A specific type of IBC packet for transferring tokens.
Bridging: Connecting Cosmos to external blockchains (e.g., Bitcoin, Ethereum) via adapter zones.
Adapter Zones: Specialized zones designed to connect to non-Tendermint chains.
Application Blockchain Interface (ABCI): The interface between Tendermint Core (consensus) and the application state machine.
AppendTx (ABCI message): Delivers transactions to the application for validation and state update.
CheckTx (ABCI message): Validates transactions for the mempool without mutating state.
Commit (ABCI message): Requests a cryptographic commitment (Merkle root) of the application state.
Light Clients: Clients that can securely verify the state without syncing the entire blockchain, by verifying validator signatures.
Long Range Attack (LRA): An attack specific to PoS where historical validators try to create a conflicting chain.
Unbonding Period: The duration validators/delegators must wait to retrieve staked Atoms, mitigating LRAs.
Weak Subjectivity: The requirement for light clients to obtain a recent, trusted state/validator set periodically.
Censorship Attacks: Malicious validators attempting to prevent specific transactions from being included.
Fork Mitigation (Reorg-proposal): Process for recovering from forks or censorship via out-of-band coordination.
Staking: Bonding Atom tokens as collateral to participate as a validator or delegator.
Slashing: The penalty (loss of stake) for validator misbehavior (e.g., double-signing).
Delegators: Atom holders who delegate their stake to validators.
Transaction Fees: Fees paid by users to validators for transaction processing.
ReserveTax: A portion of transaction fees allocated to the reserve pool.
Fundraiser: The initial distribution event for Atom tokens.
Constitution: Human-readable document outlining network policies.
Governance Proposals: On-chain mechanisms for changing parameters, coordinating upgrades, or amending the constitution (Parameter Change, Text, Bounty).
Voting (Yea/Nay/WithForce/Abstain): Options for validators/delegators on proposals.
Veto (WithForce): Mechanism allowing a 1/3+ minority to block a proposal, incurring penalties.
Merkle Tree: Cryptographic structure for efficiently verifying data inclusion.
Simple Tree: Merkle tree used for transactions within a block.
IAVL+ Tree: A balanced Merkle tree used for the application state (persistent key-value store).
Dynamic Peg: The central idea – an AMM where the price peg is not fixed but adjusts over time.
CurveCrypto Invariant: The specific mathematical formula defining the AMM curve, combining stableswap and constant-product properties.
Internal Oracle: The mechanism (specifically an EMA) used by the contract to track the perceived market price, which the dynamic peg targets.
Price Scale (p): The internal vector representing the current target prices (peg) for the assets in the pool.
Repegging Algorithm: The process and conditions under which the Price Scale (p) is updated towards the internal oracle price.
Repegging Condition: The core rule for adjusting the peg (Loss < Profit / 2), based on changes in X_cp.
X_cp (Constant-Product Value): A metric derived from the invariant D and prices p, used to quantify profit/loss for the repegging decision.
Dynamic Fees: Trading fees that adjust based on how far the current pool balances are from the equilibrium defined by the current Price Scale (p).
Concentrated Liquidity (Dynamic): Liquidity is focused around the current internal price peg, which itself can move.
Invariant (I(b')=0): The mathematical function defining the relationship between transformed balances.
Invariant D (D): Parameter defining the size/depth of the specific invariant curve (analogous to Curve V1).
Transformed Balances (b'): Asset balances normalized by the Price Scale (p).
Gamma (γ): Parameter controlling the "distance" or transition sharpness between the stableswap and constant-product components of the invariant.
Amplification Coefficient (A): Parameter influencing the curve's shape, similar to stableswap.
Exponential Moving Average (EMA): The specific mechanism used for the internal price oracle.
Newton's Method: The numerical method used to solve the invariant equation during swaps.
Fee Gamma (γ_fee): Parameter controlling the responsiveness of the dynamic fees.
Stableswap Invariant: Referenced as a basis and component for the CurveCrypto invariant.
Constant-Product (x*y=k): Referenced as a basis and the limiting behavior of the CurveCrypto invariant far from the peg.
Liquidity Providers (LPs): Key participants who benefit from the described mechanism (higher profits mentioned).
Curve Stablecoin: The subject of the design (e.g., crvUSD).
LLAMMA (Lending-Liquidating AMM Algorithm): The central, novel AMM replacing traditional liquidations.
PegKeeper: Contract responsible for maintaining the stablecoin peg using a reserve pool.
Monetary Policy: Mechanism adjusting borrow rates based on PegKeeper state to influence peg.
Controller: Contract managing rates and interacting with LLAMMA/PegKeeper.
Continuous Liquidation / Deliquidation: The core function of LLAMMA, gradually converting collateral <-> stablecoin based on price.
Special-Purpose AMM: Refers to LLAMMA's unique design focused on liquidation.
External Price Oracle (p_o): Required input for LLAMMA determining the "center" price.
Price Bands: Discrete price ranges within LLAMMA where liquidity exists.
Band Prices (p_cd, p_cu, p_↑, p_↓): Specific price points defining the operational range and edges of a band relative to the oracle price (p_o).
Virtual Balances (f, g): Parameters added to real balances in the invariant calculation, dependent on p_o.
LLAMMA Invariant: The specific constant-product-like equation (x+f)(y+g) = I governing trades within a band (where f, g, and I depend on p_o).
Liquidity Concentration (A): Parameter defining the tightness of liquidity concentration within bands.
Deposit Measure (y_0): Parameter related to the amount of deposits within a specific band.
Adiabatic Conversion: The concept that slow oracle price changes allow gradual collateral conversion within LLAMMA.
get_x_down / get_y_up: Key functions calculating the amount of stablecoin/collateral obtainable if the price moves to the band edge.
Health (Factor/Method): A measure of loan safety derived from get_x_down relative to the debt.
Base Price Multiplier: Mechanism to shift the entire LLAMMA price grid to account for accrued interest on the stablecoin debt.
Automatic Stabilizer: The combined system of PegKeeper and Monetary Policy.
Peg-Keeping Reserve: A separate liquidity pool (e.g., a Curve Stableswap pool) used by the PegKeeper.
Asymmetric Deposit/Withdrawal: PegKeeper actions adding/removing single-sided liquidity to/from the reserve pool to adjust the peg.
Mint/Burn: PegKeeper creates (mints) uncollateralized stablecoins to deposit when peg > 1 and destroys (burns) stablecoins withdrawn when peg < 1.
Stabilizer Debt (d_st): The amount of uncollateralized stablecoin minted by the PegKeeper.
Collateral: The volatile asset used to back the borrowed stablecoin (e.g., ETH).
Stablecoin: The asset designed to maintain a peg (e.g., to USD).
Liquidation Threshold: The price point below which collateral is considered at high risk.
AMM (Automated Market Maker): General category, with LLAMMA being a specific type.
Borrow Rate (r): Interest rate charged for borrowing the stablecoin, adjusted by Monetary Policy.
Curve DAO: The decentralized autonomous organization itself.
Aragon: The underlying DAO framework used for management.
CRV Token (ERC20CRV): The native governance and value accrual token.
VotingEscrow: The smart contract where CRV is locked for voting power.
Gauges: Smart contracts measuring user activity (e.g., liquidity provision) to distribute CRV inflation.
LiquidityGauge: A specific type of gauge for measuring LP token deposits.
GaugeController: The central contract managing gauge types, weights, and user vote allocations.
Minter: The contract responsible for minting CRV according to the inflation schedule.
Fee Burner: Mechanism/contract for collecting protocol fees (though burning is mentioned as one possibility).
Time-weighted Voting: Voting power (w) is proportional to both the amount of CRV locked (a) and the remaining lock time (t).
Vote-locking: The act of locking CRV tokens in VotingEscrow for a chosen duration.
Locktime: The duration for which CRV tokens are locked.
Voting Weight (w): The measure of influence in governance and gauge weight voting, derived from locked CRV and remaining locktime.
Linear Weight Decay: The voting weight decreases linearly over the lock period.
Slope/Bias Tracking: The implementation technique using linear function parameters (slope, bias) to efficiently calculate decaying voting weights without frequent updates.
Inflation Schedule: The predetermined, decreasing rate at which new CRV tokens are minted over time.
Reward Distribution (via Gauges): Directing CRV inflation to users based on their measured activity in specific gauges (e.g., LP token staking).
Gauge Weight Voting: Users allocating their voting weight (from VotingEscrow) to different gauges to determine the proportion of CRV inflation each gauge receives.
Reward Boosting: Increasing a user's CRV reward share from a gauge (up to 2.5x) based on their own locked CRV voting weight relative to the total liquidity they provide.
LP Token Staking (in Gauges): Users deposit Curve LP tokens into LiquidityGauges to earn CRV rewards.
DeFi (Decentralized Finance): The overarching domain.
Risk Assessment: The primary purpose of the guidelines.
Risk Mitigation: Strategies to reduce identified risks.
Ecosystem Stakeholders: Defined roles interacting with DeFi protocols.
Protocol Users: End-users/customers of a DeFi protocol.
Protocol Investors: Holders of governance stakes.
Protocol Operators: Entities making operational decisions.
Smart Contract Operators: Operators with privileged contract access.
Protocol Developers: Contributors to protocol code.
Smart Contracts: Core back-end software implementing protocol logic.
Blockchains: The underlying platforms where Smart Contracts run.
Oracles: Automated external information sources for Smart Contracts.
Bridges (Trusted/Trustless): Mechanisms for interoperability between blockchains.
User Interfaces (Front ends): Typically web applications for user interaction.
Wallets (Hot/Cold, Custodial/Self-Custody, Multi-Signature): Tools for managing keys and assets.
Decentralized Exchanges (DEXs): Platforms for P2P asset trading.
Automated Market Maker (AMM): Algorithm governing DEX liquidity pools.
Liquidity Pool: Pool of assets facilitating trades or lending.
Liquidity Providers: Users supplying assets to liquidity pools.
LP Tokens: Tokens representing a share in a liquidity pool.
Decentralized Lending: P2P lending/borrowing platforms.
Stablecoins (Peg, RWA-backed, Algorithmic): Tokens designed to maintain value relative to a reference asset.
Wrapped Tokens: On-chain representations of assets from another blockchain.
Yield Farming (Subsidized/Real): Earning rewards by providing assets/services to protocols.
Derivatives (DeFi): Financial instruments deriving value from underlying assets (e.g., perpetual options).
Software Risks: General risks from software design, implementation, or vulnerabilities.
Smart Contract Risk: Risks specific to Smart Contract code (e.g., reentrancy, immutability issues, upgrade risks).
Blockchain Risk: Risks from the underlying blockchain platform (stoppages, lost records, incorrect execution, 51% attacks).
User Interface Risk: Risks from frontend design, usability, accessibility, or security (e.g., confusing UX, javascript injection).
Oracle Risk: Risks from oracle data (accuracy, latency, manipulation, centralization).
Bridge Risk: Risks specific to cross-chain bridges (e.g., honeypots, counterparty/custodial risk for trusted bridges).
MEV Risk (Maliciously Extracted Value): Risks from transaction manipulation (Front-Running, Back-Running, Sandwich Attacks, Censoring).
Governance Risk: Risks from protocol decision-making processes (Rug Pull, Token Concentration, Treasury Attacks, Key Compromise, Paralysis).
Custodial Risk: Risks related to managing private keys (lost keys, theft, counterparty failure for third-party custody).
Tokenomics Risk: Risks from flawed economic design (supply distortions, inflation, limited utility, inflexibility).
Compliance and Legal Risk: Risks from violating laws/regulations (securities laws, AML/KYC, sanctions, tax).
Tax Risk: Uncertainty or adverse treatment regarding taxation of DeFi transactions.
Standards Conformance Risks: Risks from failing to meet relevant industry or accounting standards.
Accounting Conformance Risk: Challenges applying standards like GAAP/IFRS (derecognition, control definition).
Operational Accounting Risk: Inaccuracies in accounting procedures (mismatches, fraud, inconsistency, fixed valuations).
Credit Risk: Risk of loss from borrower default or insufficient collateral during liquidation (Bad Debt).
Counterparty Risk: Risk that another party in a transaction fails to meet obligations.
Market Risk: Risk of losses from changes in asset values (volatility, loss of confidence, manipulation, systemic risk).
Liquidity Risk: Risk of being unable to trade assets at fair market value due to insufficient market depth (fragmentation, liquidation failure).
Key DeFi Metrics:Total Value Locked (TVL): Value of assets held within a protocol.
Market Cap (MCAP - Circulating/Fully Diluted): Token price multiplied by supply.
Protocol Fees: Revenue generated from usage fees.
Gas Usage: Transaction fees paid by a protocol's contracts.
Impermanent Loss (Divergence Loss, LVR): Opportunity cost of providing liquidity vs. holding.
Loan to Value (LTV): Collateral requirement ratio for loans.
Best Practices: Recommended approaches for managing risks.
Risk Mitigation Strategies: Actions taken to lessen risks.
Key Information for Risk Assessment: Data needed to evaluate risks.
Operational Reporting: Historical performance and event data.
Structural Reports: Information on protocol design, governance, and policies.
Security Review / Audit (Smart Contracts): Assessment of code vulnerabilities.
EthTrust Security Levels: An EEA standard for Smart Contract reviews.
OWASP Standards: Standards for web/mobile/API security testing.
Accessibility (WCAG): Standards for user interface accessibility.
Crosschain Security Guidelines: EEA guidelines for bridge security.
Key Management Standards (CCSS, ISO27001): Standards for secure private key handling.
SOC Reports (SOC1, SOC2): Reports on service organization controls (relevant for custodians, accounting providers).
Threat Modeling: Proactive identification of potential attack vectors.
Incident Response: Procedures for handling security events.
Real-time Monitoring: Automated systems for detecting attacks or anomalies.
Stress Testing: Simulating adverse conditions to test resilience.
Internal Controls (Accounting): Procedures like reconciliation, segregation of duties.
KYC/AML: Procedures for identifying users/operators.
Time-Weighted Average Pricing (TWAP): Technique to smooth price data and detect manipulation.
Slippage Limits: User-set controls to prevent unfavorable trade execution prices.
Overcollateralization: Requiring collateral value exceeding loan value.
Rehypothecation: Re-using collateral pledged for one purpose for another.
Bug Bounty Programs: Incentives for responsible disclosure of vulnerabilities.
Responsible Disclosure: Process for reporting vulnerabilities privately first.
Elliptic Curve Cryptography (ECC): The main subject.
Public-Key Cryptography / Asymmetric Cryptography: The general category ECC belongs to.
Elliptic Curve (E): The fundamental mathematical object used.
Elliptic Curves over Finite Fields (Fq, Fp): The practical mathematical setting for ECC.
Weierstrass Equation / Form: The standard equation defining an elliptic curve (y² = x³ + ax + b).
Point at Infinity (O): The identity element in the elliptic curve group.
Mathematical Operations & Properties:
Group Law (on Elliptic Curve): The rules defining point addition.
Point Addition (⊕): The fundamental operation combining two points on the curve.
Point Doubling: Adding a point to itself (P ⊕ P).
Scalar Multiplication (kP): Repeatedly adding a point P to itself k times.
Abelian Group: The algebraic structure formed by points on the curve under addition.
Generator Point (P): A base point used to generate a cyclic subgroup via scalar multiplication.
Order (n) (of a point/subgroup): The number of distinct points generated by scalar multiplication of P before repeating (smallest k > 0 such that kP = O).
Cofactor (h): The ratio of the total number of points on the curve to the order of the subgroup generated by P.
Elliptic Curve Discrete Logarithm Problem (ECDLP): The core computational difficulty ECC relies on (given P and R=kP, find k).
Discrete Logarithm Problem (DLP): The analogous problem in standard finite fields (used in classical Diffie-Hellman).
Integer Factorization Problem: The hard problem underlying RSA security.
One-Way Function / Trapdoor Function: The conceptual basis for public-key cryptography.
Elliptic Curve Diffie-Hellman (ECDH): Key exchange protocol using ECC.
Elliptic Curve ElGamal: Encryption scheme using ECC.
Elliptic Curve Digital Signature Algorithm (ECDSA): Digital signature scheme using ECC.
Double and Add Algorithm: Efficient method for performing scalar multiplication.
Diffie-Hellman Key Exchange: The original key exchange protocol (often implemented over (Z/pZ)*).
RSA: An alternative public-key cryptosystem (mentioned for comparison).
Pollard’s Rho Algorithm: A classical algorithm for solving DLP/ECDLP (sub-exponential).
Shor’s Algorithm: A quantum algorithm that solves DLP/ECDLP/Factorization in polynomial time.
Quantum-Resistant Cryptography: Cryptosystems designed to resist attacks by quantum computers.
Isogeny-Based Cryptography: A candidate post-quantum approach using mappings between elliptic curves.
Supersingular Elliptic Curves: Specific type of curve used in isogeny-based crypto.
Isogeny: A specific type of map between elliptic curves.
Domain Parameters (ECC): The set of public values defining a specific ECC system (curve coefficients, field, base point P, order n, cofactor h).
Finite Field (Fq, Fp): The mathematical structure over which curve operations are performed.
Generator (of a finite field group): An element whose powers generate all other elements.
Digital Signatures: Cryptographic method for verifying authenticity and integrity.
Hash Functions: Used in digital signatures.
Ethereum: The name of the platform/protocol.
Smart Contract: Code implementing arbitrary rules, controlling digital assets or state transitions.
Decentralized Application (DApp): Applications built using Ethereum's platform.
State Transition System: The fundamental model of Ethereum, where transactions change the overall state.
State: The collection of all account information (balances, code, storage) at a given time.
Accounts: The fundamental objects in the Ethereum state.
Externally Owned Accounts (EOAs): Controlled by private keys.
Contract Accounts: Controlled by their associated code.
Ether: The native internal cryptocurrency of Ethereum, used for fees and as a liquidity layer.
Ethereum Virtual Machine (EVM): The execution environment for contract code.
EVM Code: The low-level, stack-based bytecode language executed by the EVM.
Turing-completeness: The property of the EVM allowing it to perform any conceivable computation (given enough resources).
Transaction (Ethereum): Signed data package from an EOA initiating a state change (value transfer or contract interaction).
Message (Ethereum): Virtual object representing calls between contracts, produced by the CALL opcode.
Gas: The unit measuring computational steps and resource usage (computation, bandwidth, storage).
Gas Price: The amount of Ether a transaction sender pays per unit of Gas.
Start Gas: The maximum Gas limit set for a transaction.
Nonce (Account): Counter preventing transaction replay for EOAs.
Storage (Contract): Persistent key/value store associated with a contract account.
Memory (EVM): Temporary, expandable byte array used during code execution.
Stack (EVM): Last-in-first-out data structure used during EVM execution.
Blockchain (Ethereum): The distributed ledger containing blocks of transactions and state roots.
Block Header: Contains metadata like timestamp, nonce, previous hash, state root, transaction root, etc.
State Root: Merkle root hash of the entire Ethereum state tree (often Patricia Tree).
Transaction Root: Merkle root hash of the transactions included in a block.
Patricia Tree (Modified Merkle Tree): Data structure used for efficiently storing and updating state and transactions.
Mining (Ethereum): The process of creating blocks using Proof-of-Work (initially proposed).
Proof-of-Work (Ethereum variant): Proposed consensus mechanism requiring memory-hard computation involving state access.
GHOST (Greedy Heaviest Observed Subtree): Modified consensus protocol incorporating "uncle" blocks to improve security on fast blockchains.
Uncle Blocks: Stale blocks included by reference in the main chain under GHOST, receiving partial rewards.
Token Systems (On-chain): Implementing custom currencies or assets via smart contracts.
Financial Derivatives: Smart contracts deriving value from external data (e.g., price feeds).
Stable-Value Currencies: Contracts aiming to peg value to external assets (requires data feed/oracle).
Identity and Reputation Systems: Using contracts for name registration or attribute tracking (like Namecoin).
Decentralized File Storage: Conceptual application using contracts and Merkle trees.
Decentralized Autonomous Organizations (DAOs): Virtual entities governed by code and member voting.
Savings Wallets: Contracts implementing custom withdrawal rules.
Crop Insurance: Example derivative based on external data (weather).
Decentralized Data Feed (SchellingCoin): Conceptual mechanism for achieving consensus on external data.
Cloud Computing (Verifiable): Using EVM for provably correct computation markets.
Peer-to-peer Gambling: Implementing betting protocols on-chain.
Prediction Markets: Markets betting on future event outcomes.
Bitcoin: The predecessor cryptocurrency, used for comparison.
UTXO (Unspent Transaction Output): Bitcoin's state model.
Scripting (Bitcoin): Bitcoin's limited, non-Turing-complete contract language.
Lack of Turing-completeness (Bitcoin Script): Key limitation overcome by Ethereum.
Value-blindness (Bitcoin Script): Inability for scripts to control the amount spent.
Lack of State (Bitcoin Script): Inability to maintain state across multiple transactions.
Blockchain-blindness (Bitcoin Script): Inability to access blockchain metadata.
Merkle Trees (Bitcoin): Used for transaction inclusion proofs.
Simplified Payment Verification (SPV): Bitcoin's light client mechanism.
Colored Coins / Metacoins: Protocols built on top of Bitcoin, discussed as alternatives.
Proof-of-Work (Bitcoin - SHA256): Bitcoin's consensus algorithm.
51% Attack: General threat to Proof-of-Work consensus
Ethereum: The name of the platform/protocol.
World State (σ): The mapping between addresses and account states, represented as a Merkle Patricia Tree.
Account State: The components of an account (nonce, balance, storageRoot, codeHash).
Externally Owned Account (EOA): Account controlled by a private key.
Contract Account: Account controlled by associated EVM code.
Address: 160-bit identifier for an account.
State Transition Function (Υ): Function defining how a transaction changes the world state.
Block: A collection of transactions, withdrawals, and a header linking to the previous block.
Blockchain: The sequence of linked blocks representing the canonical history.
Generalized Transaction Ledger: The paper's description of Ethereum's core function.
Shanghai Version: The specific protocol version described in this document.
Ethereum Virtual Machine (EVM):
Ethereum Virtual Machine (EVM): The quasi-Turing-complete state machine executing contract code.
EVM Code: Stack-based bytecode executed by the EVM.
Machine State (µ): The EVM's internal state during execution (gas, pc, memory, active words, stack).
Stack: LIFO data structure for EVM computation.
Memory: Volatile byte-addressable memory space for EVM execution.
Storage: Persistent key-value store associated with a contract account (word-addressable).
Program Counter (pc): Pointer to the current instruction in the EVM code.
Opcodes: Individual instructions executed by the EVM (e.g., ADD, SSTORE, CALL, CREATE).
Gas: Unit measuring computational cost.
Gas Cost Function (C): Defines the gas cost for each opcode.
Exceptional Halting: Conditions causing EVM execution to stop abnormally (e.g., out of gas, stack underflow, invalid jump).
Normal Halting: Controlled execution termination (STOP, RETURN, REVERT, SELFDESTRUCT).
Transaction (T): Cryptographically signed instruction from an EOA.
Transaction Types (EIP-2718): Different transaction formats (Legacy/0, EIP-2930/1, EIP-1559/2).
Nonce (Account): Transaction counter for an EOA, preventing replay.
Gas Limit (Tg): Maximum gas consumable by a transaction.
Gas Price (Tp): Fee per gas for Type 0/1 transactions.
Max Fee Per Gas (Tm): Maximum total fee per gas for Type 2 transactions.
Max Priority Fee Per Gas (Tf): Maximum miner tip per gas for Type 2 transactions.
Base Fee Per Gas (Hf): Network-defined fee per gas burned in each block (post-EIP-1559).
Effective Gas Price (p): Actual price paid per gas by the transaction signer.
Priority Fee (f): The portion of the gas fee paid to the validator/beneficiary.
Value (Tv): Amount of Ether transferred in a transaction.
Data (Td) / Init (Ti): Input data for a message call / contract creation code.
Access List (TA) (EIP-2930): List of addresses and storage keys to pre-warm, reducing gas cost.
Signature (Tr, Ts, Ty/Tw): Cryptographic proof of transaction origin (ECDSA).
Intrinsic Gas (g0): Base cost of a transaction before execution.
Accrued Substate (A): Information collected during execution (self-destruct set, logs, touched accounts, refund balance, accessed addresses/keys).
Logs (Al): Indexable event data emitted during execution.
Logs Bloom (Hb): Bloom filter for efficient log searching.
Receipts Root (He): Merkle root of transaction receipts in a block.
Transaction Root (Ht): Merkle root of transactions in a block.
Withdrawal (W): Operation transferring staked Ether from the consensus layer to the execution layer (post-Merge).
Withdrawals Root (Hw): Merkle root of withdrawals processed in a block.
Merkle Patricia Tree (Trie): Modified Radix Trie used for storing state, transactions, receipts.
Recursive Length Prefix (RLP): Serialization method used in Ethereum.
Hex-Prefix Encoding (HP): Encoding used within the Merkle Patricia Tree implementation.
Keccak-256 (KEC): The primary hash function used in Ethereum.
Consensus & Related (as referenced by Execution Layer):
Proof-of-Stake (PoS): The consensus mechanism after the Paris hard fork.
Beacon Chain: The consensus layer coordinating validators (referenced).
Paris Hard Fork: The event marking the transition to Proof-of-Stake ("The Merge").
Terminal Total Difficulty (TTD): The PoW difficulty threshold triggering the Paris hard fork.
LMD GHOST: The fork-choice rule used by the Proof-of-Stake consensus layer.
Finalized Block: A block confirmed by the consensus layer with high security guarantees.
RANDAO / prevRandao (Ha): Source of pseudo-randomness derived from the consensus layer.
Beneficiary (Hc): Address receiving priority fees (validator).
(Deprecated) Proof-of-Work (PoW): Original consensus mechanism.
(Deprecated) Ethash: Ethereum's original PoW algorithm.
(Deprecated) Difficulty (Hd): PoW difficulty measure.
(Deprecated) Ommers / Uncles (Ho): Mechanism from early PoW Ethereum (GHOST).
SECP-256k1: The elliptic curve used for signatures.
ECDSA (ECDSASIGN, ECDSARECOVER): Elliptic Curve Digital Signature Algorithm.
Precompiled Contracts: Special contracts (addresses 1-9, etc.) implementing complex cryptographic operations efficiently (e.g., ECADD, ECMUL, ECPAIRING, Blake2f, ModExp).
BLAKE2f: Compression function precompile.
Account Abstraction (referenced via EIPs): Concept aiming to make contracts first-class accounts.
Intrinsic Value (discussed): Philosophical concept contrasted with cryptocurrencies.
Smart Property: Concept of blockchain-managed ownership of physical devices.
Decentralized Autonomous Organization (DAO): Concept enabled by smart contracts.
Colored Coins / Metacoins: Early concepts for representing assets on Bitcoin.
Liquity V2: The protocol version being described.
BOLD: The new stablecoin issued by Liquity V2.
Trove: A collateralized debt position (represented as an NFT).
Collateral Assets: Assets used to back BOLD debt (initially ETH, wstETH, rETH).
Liquid Staking Tokens (LSTs): Specific type of collateral supported (wstETH, rETH).
Stability Pools (SP): Pools where users deposit BOLD to absorb liquidations (one per collateral type).
Earners: Users who deposit BOLD into Stability Pools.
Protocol-Incentivized Liquidity (PIL): Mechanism directing protocol revenue to incentivize DEX liquidity for BOLD.
LQTY Token: The token used for staking to gain voting power for PIL distribution.
Peripheral Governance: Minimal governance focused solely on PIL distribution via gauge voting.
Gauge Voting: Mechanism for LQTY stakers to direct PIL rewards.
User-set Interest Rates: Borrowers choose their own interest rate within a defined range (0.5% - 250%).
Market-Driven Interest Rates: The overall rate environment emerges from individual borrower choices influenced by redemption risk.
Interest Rate Delegation (Individual/Batch): Allowing borrowers to delegate rate management to third parties.
Redemption: Exchanging BOLD for $1 worth of collateral.
Redemption Order: Based on ascending user-set interest rates (lowest rates redeemed first).
Collateral Split (Redemption): Redeemed collateral sourced proportionally across borrow markets based on "outside" debt.
Redemption Fee: Fee paid by redeemers, kept by the affected Trove owner(s) (exponential decay + spike formula).
Separate Borrow Markets: Each collateral type has its own Troves, Stability Pool, interest rate dynamics, and risk profile.
Collateral Risk Alignment: Risk is contained within specific collateral markets; redemptions target riskier LSTs proportionally more.
Sustainable Real Yield: Yield for Earners derived directly from borrower interest payments.
Improved Peg Dynamics: Stability mechanism driven by user-set rates affecting BOLD supply/demand.
Instant Leverage: Facilitated via flash loans and PIL pools in a single transaction.
Time-based Voting Power: Voting power for PIL determined by LQTY staked amount and staking duration (without locking).
Multiple Troves per Address: Users can hold several independent debt positions.
Transferable Troves (NFTs): Debt positions represented as transferable NFTs.
Liquidation: Process of closing undercollateralized Troves.
Liquidation Threshold (Maximum LTV): The collateral ratio below which a Trove is subject to liquidation (110% for ETH, 120% for LSTs).
Liquidation Penalty: Percentage of collateral forfeited during liquidation (5%).
Just-in-Time (JIT) Liquidation: Fallback where liquidator deposits BOLD to absorb debt.
Redistribution: Fallback where liquidated debt/collateral is distributed among remaining borrowers of the same collateral type.
Total Collateralization Ratio (TCR): System-wide or per-market collateralization level.
Critical Threshold (CT): TCR level below which new debt creation is throttled (150%/160%).
Shutdown Threshold (ST): TCR level below which a borrow market is permanently shut down (110%/120%).
Collateral Shutdown: Permanent disabling of borrowing functions for a specific market upon hitting ST or oracle failure.
Single-Collateral Redemptions: Special redemption against only the shutdown collateral type, incentivized by a discount.
Opening Fee: Small fee charged when opening a Trove or increasing debt.
Premature Adjustment Fee: Fee charged for adjusting interest rates within 7 days of the last adjustment/opening/increase.
Redemption Fee: Variable fee charged on redemptions, paid to Trove owners.
Minimum Collateral Ratio (MCR): Inverse of Maximum LTV (110%/120%).
Loan-to-Value (LTV): Ratio of debt to collateral value (Max 90.91%/83.33%).
Minimum Debt: Minimum BOLD amount required to open a Trove (2000 BOLD).
Maximal Extractable Value (MEV): The central topic, the value captured via transaction manipulation.
Miner Extractable Value: The original term, now broadened.
Mechanism & Process:
Transaction Reordering: Changing the sequence of transactions within a block.
Transaction Inclusion/Exclusion: Deciding which transactions make it into a block.
Block Production: The process where miners/validators create blocks.
Mempool: Off-chain storage for unconfirmed transactions.
Gas Price: Transaction fee used as a default ordering mechanism (but exploitable).
Outsourced Block Production: The ecosystem involving Searchers, Builders, and Relayers.
Block Producers: General term for miners or validators.
Miners: Block producers in Proof-of-Work (PoW) systems.
Validators: Block producers in Proof-of-Stake (PoS) systems.
Searchers: Entities identifying MEV opportunities and creating bundles.
Builders: Entities aggregating bundles into full block payloads.
Relayers: Entities connecting builders to block producers.
Frontrunning: Placing one's own transaction before a known pending user transaction.
Backrunning: Placing one's own transaction immediately after a specific user transaction.
Sandwich Attack: Combining frontrunning and backrunning to exploit price changes caused by a user's trade.
DEX Arbitrage: Exploiting price differences between decentralized exchanges.
Liquidations (MEV context): Exploiting opportunities to liquidate DeFi loans, often involving frontrunning.
Generalized Frontrunning: Algorithmically copying/modifying profitable mempool transactions without necessarily understanding their intent.
Slippage: Negative difference between expected and executed trade price for users.
Worse Price Execution: Users receiving less favorable rates on trades.
Network Congestion: Increased gas prices due to MEV bidding wars.
Consensus Instability: Potential for block reorgs if MEV exceeds block rewards.
"Invisible Fee": Hidden costs imposed on users via MEV.
Chainlink Fair Sequencing Services (FSS): The proposed solution using decentralized oracle networks.
Decentralized Transaction Ordering: The core goal of FSS.
Secure Causal Ordering: FSS technique using transaction encryption before ordering.
Temporal Ordering (FIFO): FSS technique aiming for first-in, first-out processing.
Off-chain Transaction Collection: FSS gathers transactions before ordering.
DeFi (Decentralized Finance): The ecosystem where MEV is prevalent.
Smart Contracts: The underlying technology enabling DeFi applications.
Decentralized Exchanges (DEXs): Common venue for MEV attacks like sandwiching and arbitrage.
DeFi Lending Markets: Context for MEV related to liquidations.
Proof-of-Work (PoW) / Proof-of-Stake (PoS): Underlying blockchain consensus mechanisms.
Layer-1 / Layer-2 Networks: Blockchain layers where FSS can be applied.
Rollup Sequencers: Centralized entities in Layer-2s whose ordering power FSS aims to decentralize.
Optimistic Rollups: The central topic.
Layer-2 Scaling Solution: Its classification and primary purpose.
Off-chain Execution: Processing transactions outside the main blockchain.
Transaction Batching: Submitting multiple off-chain transactions as one to Layer-1.
"Optimistic" Assumption: Assuming off-chain transactions are valid by default.
Challenge Period: The time window during which transactions can be disputed.
Dispute: The act of challenging the validity of a transaction in a batch.
Fraud Proofs: The mechanism used to verify disputes and prove invalidity.
Fraudulent Transaction: An invalid transaction identified during a dispute.
Re-execution: Reprocessing transactions if fraud is proven.
State Adjustment: Correcting the rollup's state after a successful dispute.
Slashing / Penalty: Punishing the submitter of a fraudulent transaction by taking their stake.
Economic Stake: The collateral posted by submitters that can be slashed.
Scalability: Increasing transaction throughput.
Reduced Transaction Costs / Gas Fees: Lowering the cost of using the blockchain.
Transaction Finality: The point at which a transaction is considered irreversible (potentially delayed in Optimistic Rollups due to the challenge period).
Zero-Knowledge Rollups (zk-Rollups): A contrasting Layer-2 technology.
Validity Proofs: The mechanism used by zk-Rollups (contrast to Fraud Proofs).
Layer-1 / Base Layer / Main Blockchain / Mainnet: The underlying blockchain being scaled.
Layer-2 Protocol: The rollup system itself.
Submitter: The entity posting the transaction batch to Layer-1.
Users: Participants who can initiate disputes.
Uniswap v2: The protocol version described.
Automated Liquidity Protocol: The fundamental function.
Constant Product Formula (x * y = k): The core mathematical invariant.
Pair Contract: The core contract holding liquidity and facilitating swaps for a specific ERC20/ERC20 pair.
Factory Contract: Contract used to create and discover Pair contracts.
Router Contract: (Mentioned as separate) Contract used for user interaction, handling routing and transfers (not part of core contracts).
Core Contracts: Refers specifically to the Pair and Factory contracts, designed for minimal logic.
Liquidity Providers (LPs): Users who deposit pairs of assets into a Pair contract.
LP Tokens: Tokens representing a proportional share of a liquidity pool.
Key Features & Innovations (v2 vs v1):
ERC-20 / ERC-20 Pairs: Ability to create liquidity pools for any two ERC-20 tokens (unlike v1's ETH bridge requirement).
Time-Weighted Average Price (TWAP): The oracle provides TWAPs over chosen intervals.
Cumulative Price Accumulator: Mechanism storing the sum of prices weighted by time elapsed.
Manipulation Resistance: Achieved by accumulating prices at the beginning of blocks (or end of last trade).
Flash Swaps: Ability to receive output tokens before paying input tokens within a single atomic transaction, requiring a callback.
Protocol Fee: A switchable 5-basis-point fee (1/6th of the total 30bp fee) directed to a feeTo address, controlled by feeToSetter.
Deterministic Pair Addresses: Using CREATE2 opcode for predictable pair contract addresses.
Technical & Implementation Details:Solidity: The implementation language for v2 (v1 was Vyper).
Contract Re-architecture: Minimizing logic in the core Pair contract for enhanced security.
UQ112.112: The binary fixed-point number format used for storing prices and reserves.
Reserves Caching: Storing last known reserve values to protect the oracle.
Oracle Accumulator Storage: Using free space in reserve/timestamp storage slots for gas efficiency.
Timestamp Handling (Oracle): Using block timestamps (modulo 2^32) for TWAP calculation.
Overflow Handling (Oracle): Using extra bits in price accumulator storage for overflow.
Fee Calculation (0.30%): Fee applied on input amounts, implicitly enforced by the invariant.
Fee Collection (Protocol Fee): Collected only during mint/burn events to save gas on swaps.
sync() Function: Bail-out function to update reserves to actual balances if they diverge (e.g., due to deflationary tokens).
skim() Function: Bail-out function to withdraw excess tokens sent directly to the pair address (beyond uint112 max).
Non-standard Token Handling: Tolerating tokens without boolean returns on transfer.
Reentrancy Lock: Protection against reentrant calls, especially from ERC-777 hooks or flash swap callbacks.
Liquidity Token Initialization: Initial minting based on the geometric mean (sqrt(x * y)) of deposited amounts.
Minimum Liquidity Burn: Burning the first MINIMUM_LIQUIDITY (1e-15) LP tokens to increase costs for certain attacks.
WETH Requirement: Native ETH must be wrapped into WETH for use in v2 pairs.
Meta Transactions (LP Tokens): Support for gasless approvals/transfers of LP tokens via the permit function (EIP-712).
Uniswap v3: The protocol version described.
Concentrated Liquidity: The defining feature – LPs provide liquidity within specific price ranges.
Automated Market Maker (AMM): The general category.
Constant Product Market Maker (CFMM): The underlying formula (x*y=k) still used conceptually within active ranges.
Position: A specific instance of liquidity provided within a defined price range (tickLower, tickUpper) by an LP.
Non-Fungible Liquidity: Liquidity positions are unique and not interchangeable like V1/V2 LP tokens (often represented by NFTs externally).
Capital Efficiency: A primary goal achieved through concentrated liquidity.
Virtual Reserves: Theoretical reserves if liquidity were unbounded, used for pricing logic within ticks.
Real Reserves: The actual token balances held by a position or the pool.
Liquidity (L): The core measure representing the amount of virtual liquidity (sqrt(k) equivalent).
Multiple Pools Per Pair: Allowing different fee tiers for the same token pair.
Flexible Fees / Fee Tiers: Multiple fixed fee levels (e.g., 0.05%, 0.30%, 1%) set per pool at creation.
Improved Price Oracle:Time-Weighted Average Price (TWAP) Oracle: V3's oracle mechanism.
Geometric Mean TWAP: Calculated using the sum of log prices (specifically tick indices).
On-chain Oracle Observations: Storing historical accumulator values directly in the pool contract.
Log Price Accumulator: Tracks sum(log_1.0001(P) * time_elapsed).
Liquidity Oracle: Accumulator tracking time-weighted inverse liquidity (secondsPerLiquidityCumulative), useful for external contracts (e.g., liquidity mining).
Protocol Fee Governance: Ability for UNI governance to set a fraction of LP fees to be diverted to the protocol (per pool).
Ticks: Discrete price points based on powers of 1.0001.
Tick Index (i_c): Integer representing the current price tick.
tickLower, tickUpper: The lower and upper tick indices defining a position's range.
Square Root Price (sqrt(P) or sqrtPriceX96): The price representation used internally for computational efficiency and precision (Q64.96 fixed point).
tickSpacing: Parameter defining the granularity of initializable ticks for a pool (linked to fee tier).
Global State: Core pool variables (e.g., liquidity, sqrtPriceX96, tick, feeGrowthGlobal{0,1}).
Tick-Indexed State: Data stored per initialized tick (e.g., liquidityNet, liquidityGross, feeGrowthOutside{0,1}, secondsOutside).
liquidityNet: Net change in L when the tick is crossed.
liquidityGross: Gross liquidity referencing the tick (for initialization tracking).
feeGrowthOutside: Fees accumulated outside the tick range.
Position-Indexed State: Data stored per liquidity position (e.g., liquidity, feeGrowthInside{0,1}Last).
feeGrowthInsideLast: Snapshot of fees earned inside the range when the position was last touched.
Initialized Tick Bitmap (tickBitmap): Optimization for efficiently finding the next active tick during swaps.
Non-Compounding Fees: Fees are collected as tokens, not automatically reinvested into the pool.
Factory Contract (V3): Contract for deploying Pair contracts.
Pair Contract (V3): The core contract managing a specific token pair pool.
UQ112.112 / uint112 Reserves: (Context from V2, partially relevant for understanding fixed-point choices, though V3 uses different internal representation).
CREATE2 Opcode: Used for deterministic pair addresses.
Maker Protocol: The overall system, also known as the Multi-Collateral Dai (MCD) system.
Multi-Collateral Dai (MCD): The current system allowing multiple collateral types (distinct from SCD).
Single-Collateral Dai (SCD) / Sai: The original version using only ETH as collateral.
Decentralized Autonomous Organization (DAO): The organizational structure of MakerDAO.
Ethereum: The underlying blockchain platform.
Dai (Stablecoin): The decentralized, collateral-backed stablecoin soft-pegged to the US Dollar.
MKR (Governance Token): The token used for voting and recapitalization in the Maker Protocol.
Maker Vaults: Smart contracts where users lock collateral to generate Dai (replaces the term CDP).
Collateral Assets: Ethereum-based assets approved by Maker Governance to back Dai.
Oracles / Oracle Feeds: Provide real-time price information for collateral assets.
OracleDai Foundation: Independent entity safeguarding intangible assets (trademarks, code copyrights).
Soft-Peg: Maintaining a value close to a target (e.g., 1 Dai ≈ 1 USD) via economic incentives rather than a hard link.
Decentralized / Unbiased: Key philosophical properties of Dai and the Maker Protocol.
Overcollateralization: Requiring the value of locked collateral to exceed the value of generated Dai.
Non-custodial: Users retain control over their Vaults unless liquidated.
"""

def data_synth():
    prompt = f"""

  "prompt_instructions": 
    "task": "Generate synthetic training data for fine-tuning a ModernBERT model. This model will act as a query router for a multi-source RAG chatbot focused on cryptocurrency and DeFi.",
    "objective": "The router must classify incoming user queries ('text') into one of three categories ('label') based on the optimal information source required to answer the query.",
    "labels": 
      "1": "Native LLM Response: Query can be answered well by the base LLM's general knowledge. Includes greetings, conversational chat, general knowledge questions (even basic crypto concepts), coding help, creative tasks, math, etc.",
      "2": "Vectorstore-based RAG (Corpus-Specific): Query requires specific information, details, explanations, comparisons, or mechanisms found *only* within a predefined corpus of cryptocurrency/DeFi documents. The answer is likely contained within technical whitepapers or documentation.",
      "3": "Web Search-based Augmentation: Query requires real-time data (prices, volumes, gas fees), recent news/events, information about protocols/coins *not* covered in the corpus, current market sentiment, or very recent developments/regulations."
    ,
    "corpus_context_keywords": {document_keywords}],
    "generation_guidelines": 
      "Generate a diverse range of user queries reflecting realistic interactions.",
      "Ensure Label 2 queries specifically target details that would likely *only* be found in documents covering the `corpus_context_keywords`. Avoid generic crypto questions for Label 2.",
      "Ensure Label 3 queries explicitly ask for current/real-time information or relate to topics clearly outside the `corpus_context_keywords`.",
      "Vary sentence structure, question types (what, how, why, compare, list, summarize, explain), and phrasing.",
      "Include some potentially ambiguous queries that test the boundaries between categories, but ensure the assigned label is the most logical primary source.",
      "Avoid simple repetition; aim for unique semantic meaning in each query.",
      "Generate a substantial number of examples, aiming for a balanced distribution across the three labels, but allowing for natural variation.",
      "The output should be a single JSON object where keys are the synthetic user queries ('text') and values are the corresponding integer labels (1, 2, or 3)."
    ],
    "output_format_example": 
      "What is the constant product formula used by Uniswap V2?": 2,
      "Tell me a joke about Bitcoin.": 1,
      "What is the current price of Solana?": 3,
      "How do flash loans work according to the Aave documentation?": 2,
      "Explain blockchain simply.": 1,
      "Latest news on Ethereum merge updates?": 3 
  ,
  "request": "Generate a JSON object (response starts with , ends with ) containing at least 1000 key-value pairs (dont stop until u get them) following the structure defined in 'output_format_example', based on all the instructions provided."
"""
    return json.loads(client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt]).text.replace('```json', '').replace('```', ''))

database = pd.DataFrame({'Sentence': [], 'Label': []})
for i in range(30):
    try:
        new_data = data_synth()  # dict
        for key in new_data.keys():
            new_row = pd.DataFrame({'Sentence': [key], 'Label': [new_data[key]]})
            database = pd.concat([database, new_row], ignore_index=True)
    except:
        pass

database.to_excel('synthetic_dataset_for_router__.xlsx')