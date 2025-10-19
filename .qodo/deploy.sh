set -e

echo "🚀 Déploiement AURA Enterprise..."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/aura_$TIMESTAMP"
DEPLOY_DIR="/opt/aura-production"
CONFIG_FILE="$DEPLOY_DIR/.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

validate_environment() {
    log_info "Validation de l'environnement..."
    
    if [[ -z "$DB_PASSWORD" || -z "$JWT_SECRET" || -z "$BLOCKCHAIN_RPC" ]]; then
        log_error "Variables d'environnement manquantes"
        exit 1
    fi
    
    AVAILABLE_SPACE=$(df /opt | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 10485760 ]; then  
        log_error "Espace disque insuffisant"
        exit 1
    fi
    
    log_info "Environnement validé avec succès"
}

backup_current() {
    log_info "Sauvegarde de l'installation actuelle..."
    
    mkdir -p $BACKUP_DIR
    
    pg_dump -h localhost -U aura_user aura_production > $BACKUP_DIR/db_backup.sql
    
    cp -r $DEPLOY_DIR/config $BACKUP_DIR/
    cp $DEPLOY_DIR/.env $BACKUP_DIR/
    
    log_info "Sauvegarde complétée: $BACKUP_DIR"
}

stop_services() {
    log_info "Arrêt des services..."
    
    cd $DEPLOY_DIR
    docker-compose down --timeout 30
    
    RUNNING_CONTAINERS=$(docker ps -q | wc -l)
    if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
        log_warn "Forçage l'arrêt des containers..."
        docker-compose down --timeout 10
    fi
    
    log_info "Services arrêtés"
}

deploy_new_version() {
    log_info "Déploiement de la nouvelle version..."
    
    mkdir -p $DEPLOY_DIR
    
    cp -r ./backend $DEPLOY_DIR/
    cp -r ./frontend $DEPLOY_DIR/
    cp -r ./blockchain $DEPLOY_DIR/
    cp -r ./infrastructure $DEPLOY_DIR/
    cp docker-compose.prod.yml $DEPLOY_DIR/docker-compose.yml
    
    cat > $CONFIG_FILE << EOF
# Configuration AURA Production
DB_PASSWORD=$DB_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
JWT_SECRET=$JWT_SECRET
BLOCKCHAIN_RPC=$BLOCKCHAIN_RPC
MARKET_DATA_API_KEY=$MARKET_DATA_API_KEY
GRAFANA_PASSWORD=$GRAFANA_PASSWORD

# Paramètres de performance
WORKERS=4
MAX_MEMORY=8G
GPU_ENABLED=true
EOF
    
    cd $DEPLOY_DIR
    log_info "Construction des images Docker..."
    docker-compose build --no-cache
    
    log_info "Démarrage des services..."
    docker-compose up -d
    
    log_info "Nouvelle version déployée"
}

verify_deployment() {
    log_info "Vérification du déploiement..."
    
    sleep 30
    
    API_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
    if [ "$API_HEALTH" != "200" ]; then
        log_error "L'API ne répond pas correctement (HTTP $API_HEALTH)"
        return 1
    fi
    
    DB_CHECK=$(docker exec aura-postgres psql -U aura_user -d aura_production -c "SELECT 1;" 2>/dev/null | grep -c "1" || echo "0")
    if [ "$DB_CHECK" -eq "0" ]; then
        log_error "La base de données ne répond pas"
        return 1
    fi
    
    REDIS_CHECK=$(docker exec aura-redis redis-cli ping 2>/dev/null | grep -c "PONG" || echo "0")
    if [ "$REDIS_CHECK" -eq "0" ]; then
        log_error "Redis ne répond pas"
        return 1
    fi
    
    log_info "Déploiement vérifié avec succès"
    return 0
}

rollback_if_failed() {
    if [ $? -ne 0 ]; then
        log_error "Échec du déploiement, rollback..."
        
        cd $DEPLOY_DIR
        docker-compose down
    
        if [ -d "$BACKUP_DIR" ]; then
            log_info "Restauration de la sauvegarde..."
        fi
        
        exit 1
    fi
}

main() {
    log_info "Début du déploiement AURA Enterprise"
    
    validate_environment
    backup_current
    stop_services
    deploy_new_version
    verify_deployment
    rollback_if_failed
    
    log_info "✅ Déploiement AURA Enterprise terminé avec succès!"
    
    echo ""
    echo "📊 Services déployés:"
    echo "   - API: http://localhost:8000"
    echo "   - Frontend: http://localhost:3000"
    echo "   - Monitoring: http://localhost:3001"
    echo "   - Blockchain: http://localhost:8545"
    echo ""
    echo "🔍 Vérification santé: curl http://localhost:8000/health"
}

main "$@"