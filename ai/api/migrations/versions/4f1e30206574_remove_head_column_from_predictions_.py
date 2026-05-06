"""remove head column from predictions_heads

Revision ID: 4f1e30206574
Revises: 9ea8eb380bbe
Create Date: 2026-04-27 13:49:03.954898

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4f1e30206574'
down_revision: Union[str, Sequence[str], None] = '9ea8eb380bbe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('predictions_heads', schema=None) as batch_op:
        batch_op.drop_constraint('uq_pred_head_veiculo_target_head', type_='unique')
        batch_op.drop_column('head')
        batch_op.create_unique_constraint('uq_pred_head_veiculo_target_dt', ['veiculo_id', 'target', 'dt_inicio'])


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('predictions_heads', schema=None) as batch_op:
        batch_op.drop_constraint('uq_pred_head_veiculo_target_dt', type_='unique')
        batch_op.add_column(sa.Column('head', sa.INTEGER(), nullable=False))
        batch_op.create_unique_constraint('uq_pred_head_veiculo_target_head', ['veiculo_id', 'target', 'head'])
